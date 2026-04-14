//! Binary I/O utilities.

use std::io::{Error, Read, Write};
use std::path::PathBuf;

use bincode::{de::read::Reader, enc::write::Writer};

pub fn file_path_in_crate(file_name: &str) -> PathBuf {
    let crate_root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(crate_root).join(file_name)
}

/// Thin wrapper around `std::fs::File` that implements bincode's `Writer` trait for binary output.
pub struct BinFileWriter(std::fs::File);

impl BinFileWriter {
    /// Create a new instance.
    pub fn new(path: PathBuf) -> Result<Self, Error> {
        std::fs::File::create(path).map(Self)
    }
}

impl Writer for BinFileWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), bincode::error::EncodeError> {
        self.0
            .write(bytes)
            .map(|_| ())
            .map_err(|_| bincode::error::EncodeError::Other("IO write error."))
    }
}

/// Thin wrapper around `std::fs::File` that implements bincode's `Reader` trait for binary input.
pub struct BinFileReader(std::fs::File);

impl BinFileReader {
    /// Create a new instance.
    pub fn new(path: PathBuf) -> Result<Self, Error> {
        std::fs::File::open(path).map(Self)
    }
}

impl Reader for BinFileReader {
    fn read(&mut self, bytes: &mut [u8]) -> Result<(), bincode::error::DecodeError> {
        self.0
            .read(bytes)
            .map(|_| ())
            .map_err(|_| bincode::error::DecodeError::Other("IO read error."))
    }
}
