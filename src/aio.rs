//! Overlapped (asynchronous) unbuffered reads on Windows through an I/O
//! completion port: one thread can keep many requests in flight, so reads can
//! be small (cheap to stop) without starving the drive of queue depth.

use std::fs::File;
use std::io;
use std::os::windows::io::AsRawHandle;

type Handle = *mut core::ffi::c_void;

/// Win32 OVERLAPPED: kernel bookkeeping, the file offset, an optional event.
#[repr(C)]
pub struct Overlapped {
    internal: usize,
    internal_high: usize,
    offset: u32,
    offset_high: u32,
    event: Handle,
}

impl Overlapped {
    pub fn at(offset: u64) -> Overlapped {
        Overlapped {
            internal: 0,
            internal_high: 0,
            offset: offset as u32,
            offset_high: (offset >> 32) as u32,
            event: std::ptr::null_mut(),
        }
    }
}

#[link(name = "kernel32")]
extern "system" {
    fn CreateIoCompletionPort(file: Handle, port: Handle, key: usize, threads: u32) -> Handle;
    fn GetQueuedCompletionStatus(port: Handle, bytes: *mut u32, key: *mut usize, ov: *mut *mut Overlapped, ms: u32) -> i32;
    fn PostQueuedCompletionStatus(port: Handle, bytes: u32, key: usize, ov: *mut Overlapped) -> i32;
    fn ReadFile(file: Handle, buf: *mut u8, len: u32, read: *mut u32, ov: *mut Overlapped) -> i32;
    fn CancelIoEx(file: Handle, ov: *mut Overlapped) -> i32;
    fn CloseHandle(h: Handle) -> i32;
    fn GetLastError() -> u32;
}

const ERROR_IO_PENDING: u32 = 997;
const INFINITE: u32 = u32::MAX;

/// Open `path` for unbuffered overlapped reads.
pub fn open(path: &std::path::Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED).open(path)
}

/// A completion port that files are attached to.
pub struct Port(Handle);
unsafe impl Send for Port {}
unsafe impl Sync for Port {}

/// One finished request: bytes moved, the request's OVERLAPPED, and whether
/// it succeeded (false also when cancelled).
pub struct Done {
    pub bytes: usize,
    pub ov: *mut Overlapped,
    pub ok: bool,
}

impl Port {
    pub fn new() -> io::Result<Port> {
        let h = unsafe { CreateIoCompletionPort(-1isize as Handle, std::ptr::null_mut(), 0, 1) };
        if h.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Port(h))
    }

    /// Route `file`'s completions here, tagged with `key`.
    pub fn attach(&self, file: &File, key: usize) -> io::Result<()> {
        let h = unsafe { CreateIoCompletionPort(file.as_raw_handle() as Handle, self.0, key, 0) };
        if h.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Start reading `len` bytes at the OVERLAPPED's offset into `buf`.
    ///
    /// # Safety
    /// `buf` (sector-aligned, `len` a multiple of the sector size) and `ov`
    /// must stay valid until the request's completion is taken from `wait`.
    pub unsafe fn read(&self, file: &File, buf: *mut u8, len: usize, ov: *mut Overlapped) -> io::Result<()> {
        if ReadFile(file.as_raw_handle() as Handle, buf, len as u32, std::ptr::null_mut(), ov) == 0 {
            let e = GetLastError();
            if e != ERROR_IO_PENDING {
                return Err(io::Error::from_raw_os_error(e as i32));
            }
        }
        Ok(())
    }

    /// Ask the OS to drop a request still in flight; its completion still
    /// arrives (as failed, or done if it was too late to stop).
    pub fn cancel(&self, file: &File, ov: *mut Overlapped) {
        unsafe { CancelIoEx(file.as_raw_handle() as Handle, ov) };
    }

    /// Wake a `wait` with a null request (a doorbell for new work).
    pub fn ring(&self) {
        unsafe { PostQueuedCompletionStatus(self.0, 0, usize::MAX, std::ptr::null_mut()) };
    }

    /// Next completion; `None` for a doorbell.
    pub fn wait(&self) -> Option<Done> {
        let (mut bytes, mut key, mut ov) = (0u32, 0usize, std::ptr::null_mut());
        let ok = unsafe { GetQueuedCompletionStatus(self.0, &mut bytes, &mut key, &mut ov, INFINITE) } != 0;
        if ov.is_null() {
            return None;
        }
        let _ = key;
        Some(Done { bytes: bytes as usize, ov, ok })
    }
}

impl Drop for Port {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.0) };
    }
}

/// Throughput of random reads of `size` bytes with `depth` in flight.
pub fn bench(paths: &[std::path::PathBuf], size: usize, depth: usize, secs: f64) -> io::Result<f64> {
    let files: Vec<File> = paths.iter().map(|p| open(p)).collect::<io::Result<_>>()?;
    let lens: Vec<u64> = files.iter().map(|f| f.metadata().map(|m| m.len())).collect::<io::Result<_>>()?;
    let port = Port::new()?;
    for (i, f) in files.iter().enumerate() {
        port.attach(f, i)?;
    }
    let layout = std::alloc::Layout::from_size_align(size, 4096).unwrap();
    let bufs: Vec<*mut u8> = (0..depth).map(|_| unsafe { std::alloc::alloc(layout) }).collect();
    let mut ovs: Vec<Box<Overlapped>> = (0..depth).map(|_| Box::new(Overlapped::at(0))).collect();
    let ov_ptrs: Vec<*mut Overlapped> = ovs.iter_mut().map(|b| &mut **b as *mut Overlapped).collect();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let issue = |slot: usize, port: &Port, next: &mut dyn FnMut(u64) -> u64| -> io::Result<()> {
        let f = next(files.len() as u64) as usize;
        let blocks = (lens[f] / size as u64).max(2) - 1;
        let off = next(blocks) * size as u64;
        let ov = ov_ptrs[slot];
        unsafe {
            *ov = Overlapped::at(off);
            port.read(&files[f], bufs[slot], size, ov)
        }
    };
    let t0 = std::time::Instant::now();
    for slot in 0..depth {
        issue(slot, &port, &mut next)?;
    }
    let mut total = 0usize;
    let mut inflight = depth;
    while inflight > 0 {
        let Some(d) = port.wait() else { continue };
        inflight -= 1;
        total += d.bytes;
        if t0.elapsed().as_secs_f64() < secs {
            let slot = ov_ptrs.iter().position(|&o| o == d.ov).unwrap();
            issue(slot, &port, &mut next)?;
            inflight += 1;
        }
    }
    let gbs = total as f64 / t0.elapsed().as_secs_f64() / 1e9;
    for b in bufs {
        unsafe { std::alloc::dealloc(b, layout) };
    }
    Ok(gbs)
}
