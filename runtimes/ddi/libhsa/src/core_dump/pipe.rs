// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Bounded core-pipe output with stable helper identity, even when the
//! application has its own process-wide child reaper.

use std::io::{self, Write};
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixDatagram;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Constructs a diagnostic error outside the pre-exec child.
fn native(message: &str) -> io::Error {
    io::Error::other(message)
}

/// Opens identity in the child itself, before arbitrary helper code can exit.
/// Only raw Linux syscalls and initialized stack storage run after fork.
fn send_own_pidfd(socket: RawFd) -> io::Result<()> {
    // SAFETY: No allocation or application lock is used after fork. The kernel
    // synchronously copies these stack records. A successful SCM_RIGHTS send
    // retains the pidfd independently of process exit and external reaping.
    unsafe {
        let fd = libc::syscall(libc::SYS_pidfd_open, libc::getpid(), 0) as i32;
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // Require stable group signaling before exec can create descendants.
        if libc::syscall(
            libc::SYS_pidfd_send_signal,
            fd,
            0,
            std::ptr::null::<libc::siginfo_t>(),
            libc::PIDFD_SIGNAL_PROCESS_GROUP,
        ) != 0
        {
            let error = io::Error::last_os_error();
            libc::close(fd);
            return Err(error);
        }
        let mut byte = 1_u8;
        let mut vector = libc::iovec {
            iov_base: (&raw mut byte).cast(),
            iov_len: 1,
        };
        let mut control = [0_usize; 4];
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &raw mut vector;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = libc::CMSG_SPACE(size_of::<i32>() as u32) as usize;
        let cmsg = libc::CMSG_FIRSTHDR(&raw const message);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(size_of::<i32>() as u32) as usize;
        libc::CMSG_DATA(cmsg).cast::<i32>().write(fd);
        let sent = libc::sendmsg(
            socket,
            &raw const message,
            libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
        );
        let error = io::Error::last_os_error();
        libc::close(fd);
        if sent == 1 { Ok(()) } else { Err(error) }
    }
}

/// Receives the one descriptor queued by the successfully spawned helper.
fn receive_pidfd(socket: RawFd) -> io::Result<OwnedFd> {
    // SAFETY: The socket is private to this spawn and its peer sent precisely
    // one descriptor. All receive buffers are initialized and suitably aligned.
    unsafe {
        let mut byte = 0_u8;
        let mut vector = libc::iovec {
            iov_base: (&raw mut byte).cast(),
            iov_len: 1,
        };
        let mut control = [0_usize; 4];
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &raw mut vector;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = size_of_val(&control);
        if libc::recvmsg(
            socket,
            &raw mut message,
            libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC,
        ) < 0
        {
            return Err(io::Error::last_os_error());
        }
        let cmsg = libc::CMSG_FIRSTHDR(&raw const message);
        if cmsg.is_null()
            || (*cmsg).cmsg_level != libc::SOL_SOCKET
            || (*cmsg).cmsg_type != libc::SCM_RIGHTS
            || (*cmsg).cmsg_len != libc::CMSG_LEN(size_of::<i32>() as u32) as usize
        {
            return Err(native("missing core handler pidfd"));
        }
        let fd = OwnedFd::from_raw_fd(libc::CMSG_DATA(cmsg).cast::<i32>().read());
        if byte != 1 || message.msg_flags & libc::MSG_CTRUNC != 0 {
            return Err(native("invalid core handler pidfd message"));
        }
        Ok(fd)
    }
}

/// Creates a separate handler group and receives its pre-exec identity.
fn spawn_pipe(args: &[String]) -> io::Result<(std::process::ChildStdin, OwnedFd)> {
    let (parent, child_socket) = UnixDatagram::pair()?;
    let socket = child_socket.as_raw_fd();
    let mut command = Command::new(&args[0]);
    command
        .args(&args[1..])
        .stdin(Stdio::piped())
        .process_group(0);
    // SAFETY: The closure captures only a descriptor retained through spawn.
    // Its implementation uses raw syscalls/stack data and does not allocate or
    // take locks. The newly created socket has room for its single datagram.
    unsafe {
        command.pre_exec(move || send_own_pidfd(socket));
    }
    let mut child = command.spawn()?;
    drop(child_socket); // Release a descriptor slot before receiving SCM_RIGHTS.
    let pidfd = receive_pidfd(parent.as_raw_fd())?;
    let pipe = child
        .stdin
        .take()
        .ok_or_else(|| native("missing pipe stdin"))?;
    // Child::drop neither signals nor reaps; all subsequent control uses pidfd.
    Ok((pipe, pidfd))
}

/// Signals the original process group without a numeric PID lookup.
fn signal_group(pidfd: &OwnedFd, signal: i32) -> io::Result<()> {
    // SAFETY: The owned descriptor identifies the original process/group; the
    // null siginfo requests ordinary signal delivery with no borrowed pointers.
    if unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd.as_raw_fd(),
            signal,
            std::ptr::null::<libc::siginfo_t>(),
            libc::PIDFD_SIGNAL_PROCESS_GROUP,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Reaps only the original child; an application reaper produces ECHILD.
fn child_exited(pidfd: &OwnedFd, nonblocking: bool) -> io::Result<Option<bool>> {
    // SAFETY: waitid writes one initialized record and uses the retained pidfd,
    // so a reused numeric PID can never select another application child.
    unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        if libc::waitid(
            libc::P_PIDFD,
            pidfd.as_raw_fd() as u32,
            &raw mut info,
            libc::WEXITED | if nonblocking { libc::WNOHANG } else { 0 },
        ) != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok((info.si_pid() != 0).then(|| info.si_code == libc::CLD_EXITED && info.si_status() == 0))
    }
}

/// Splits a pipe command using `ROCr`'s double-quote and backslash rules.
pub(super) fn command_args(command: &str) -> io::Result<Vec<String>> {
    let mut args = Vec::new();
    let mut word = String::new();
    let (mut quoted, mut escaped, mut started) = (false, false, false);
    for ch in command.chars() {
        if escaped {
            word.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
            started = true;
        } else if ch == '"' {
            quoted = !quoted;
            started = true;
        } else if ch.is_ascii_whitespace() && !quoted {
            if started {
                args.push(std::mem::take(&mut word));
                started = false;
            }
        } else {
            word.push(ch);
            started = true;
        }
    }
    if escaped || quoted {
        return Err(native("unterminated core pipe command"));
    }
    if started {
        args.push(word);
    }
    if args.first().is_none_or(String::is_empty) {
        return Err(native("empty core pipe command"));
    }
    Ok(args)
}

/// Keeps a broken pipe from replacing the intended SIGABRT with SIGPIPE.
struct BlockSigpipe {
    /// Calling thread's exact signal mask, restored after consuming our signal.
    previous: libc::sigset_t,
    /// Set containing only SIGPIPE.
    set: libc::sigset_t,
}

impl BlockSigpipe {
    /// Blocks SIGPIPE only on the dump-writing thread.
    fn new() -> io::Result<Self> {
        // SAFETY: Both sets are initialized before use; pthread_sigmask writes
        // the old mask and copies the new one synchronously.
        unsafe {
            let mut set = std::mem::zeroed();
            let mut previous = std::mem::zeroed();
            libc::sigemptyset(&raw mut set);
            libc::sigaddset(&raw mut set, libc::SIGPIPE);
            let status = libc::pthread_sigmask(libc::SIG_BLOCK, &raw const set, &raw mut previous);
            if status != 0 {
                return Err(io::Error::from_raw_os_error(status));
            }
            Ok(Self { previous, set })
        }
    }
}
impl Drop for BlockSigpipe {
    fn drop(&mut self) {
        // SAFETY: The saved masks are initialized. Zero-timeout sigtimedwait
        // consumes a pending signal without waiting; no global disposition changes.
        unsafe {
            if libc::sigismember(&raw const self.previous, libc::SIGPIPE) == 0 {
                let timeout = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                };
                libc::sigtimedwait(
                    &raw const self.set,
                    std::ptr::null_mut(),
                    &raw const timeout,
                );
            }
            libc::pthread_sigmask(
                libc::SIG_SETMASK,
                &raw const self.previous,
                std::ptr::null_mut(),
            );
        }
    }
}

/// A bounded nonblocking stream to a potentially unresponsive user handler.
struct PipeWriter {
    /// Child stdin owned until all bytes have been sent.
    pipe: std::process::ChildStdin,
    /// Shared write-and-exit deadline.
    deadline: Instant,
}
impl Write for PipeWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        loop {
            if Instant::now() >= self.deadline {
                return Err(io::ErrorKind::TimedOut.into());
            }
            match self.pipe.write(bytes) {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let mut poll = libc::pollfd {
                        fd: self.pipe.as_raw_fd(),
                        events: libc::POLLOUT,
                        revents: 0,
                    };
                    // SAFETY: One live poll record and a bounded millisecond timeout.
                    if unsafe { libc::poll(&raw mut poll, 1, 20) } < 0 {
                        let error = io::Error::last_os_error();
                        if error.kind() != io::ErrorKind::Interrupted {
                            return Err(error);
                        }
                    }
                }
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Executes an explicitly requested handler with a bounded write/exit deadline.
pub(super) fn write_pipe(
    command: &str,
    emit: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> io::Result<()> {
    write_pipe_with_timeout(command, Duration::from_secs(5), emit)
}

/// Owns a separate handler process group until consumption succeeds or cleanup.
pub(super) fn write_pipe_with_timeout(
    command: &str,
    timeout: Duration,
    emit: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> io::Result<()> {
    let args = command_args(command)?;
    let (pipe, pidfd) = spawn_pipe(&args)?;
    let result = (|| {
        // SAFETY: fcntl reads/sets descriptor flags without consuming this live fd.
        let flags = unsafe { libc::fcntl(pipe.as_raw_fd(), libc::F_GETFL) };
        if flags < 0
            || unsafe { libc::fcntl(pipe.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
        {
            return Err(io::Error::last_os_error());
        }
        let deadline = Instant::now() + timeout;
        {
            let _sigpipe = BlockSigpipe::new()?;
            let mut writer = PipeWriter { pipe, deadline };
            emit(&mut writer)?;
        }
        loop {
            match child_exited(&pidfd, true) {
                Ok(Some(true)) => {
                    return Ok(());
                }
                Ok(Some(false)) => return Err(native("core pipe handler failed")),
                Ok(None) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
            if Instant::now() >= deadline {
                return Err(io::ErrorKind::TimedOut.into());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    })();
    if result.is_err() {
        // The pidfd was created by the child before exec, and received through
        // SCM_RIGHTS. Even a process-wide reaper cannot redirect this signal.
        let _ = signal_group(&pidfd, libc::SIGKILL);
        loop {
            match child_exited(&pidfd, false) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                _ => break,
            }
        }
    }
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pidfd_group_cleanup_survives_reaping_of_the_original_leader() {
        let path =
            std::env::temp_dir().join(format!("libhsa-core-pidfd-{}.pid", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let args = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            format!("sleep 60 & echo $! > {}; exit 0", path.display()),
        ];
        let (pipe, pidfd) = spawn_pipe(&args).unwrap();
        drop(pipe);
        // Deliberately reap the leader before cleanup, as an application reaper
        // may do. Its descriptor still names the original descendant group.
        assert_eq!(child_exited(&pidfd, false).unwrap(), Some(true));
        signal_group(&pidfd, libc::SIGKILL).unwrap();
        assert_eq!(
            child_exited(&pidfd, true).unwrap_err().raw_os_error(),
            Some(libc::ECHILD)
        );
        let pid: u32 = std::fs::read_to_string(&path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        std::fs::remove_file(path).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"));
            if stat
                .as_ref()
                .map_or(true, |stat| stat.split_whitespace().nth(2) == Some("Z"))
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "original helper group still running"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn missing_pidfd_handoff_is_nonblocking_and_never_uses_a_numeric_pid() {
        let (parent, _child) = UnixDatagram::pair().unwrap();
        assert_eq!(
            receive_pidfd(parent.as_raw_fd()).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
}
