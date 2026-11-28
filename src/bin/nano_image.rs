//! Encode image files into Nano embeddings in command-line order.
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use visers_nano::{
    image_io::{encode_files, DecodeMode, FileEmbedding},
    NanoEncoder,
};

const USAGE: &str = "\
Usage: nano_image [--full] [--jobs N] [--profile] [-o vectors.f32] [--] IMAGE...
  --full       Decode the full image before resizing (default: fast thumbnail decoding)
  --jobs N     Maximum concurrent files (default: available logical CPUs)
  --profile    Write per-file stage times and batch throughput to stderr
  -o, --output Write consecutive little-endian f32 vectors to a file
               By default, print one JSON array per input to stdout";

struct Args {
    paths: Vec<PathBuf>,
    output: Option<PathBuf>,
    jobs: NonZeroUsize,
    mode: DecodeMode,
    profile: bool,
}

impl Args {
    fn parse() -> Result<Option<Self>, Box<dyn Error>> {
        let mut args = Self {
            paths: Vec::new(),
            output: None,
            jobs: std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
            mode: DecodeMode::Fast,
            profile: false,
        };
        let mut inputs = env::args_os().skip(1);
        while let Some(arg) = inputs.next() {
            match arg.to_str() {
                Some("-h" | "--help") => return Ok(None),
                Some("--") => {
                    args.paths.extend(inputs.map(PathBuf::from));
                    break;
                }
                Some("--full") => args.mode = DecodeMode::Full,
                Some("--profile") => args.profile = true,
                Some("-j" | "--jobs") => {
                    args.jobs = inputs
                        .next()
                        .and_then(|n| n.to_str()?.parse().ok())
                        .ok_or("--jobs requires a positive integer")?;
                }
                Some("-o" | "--output") => {
                    args.output = Some(inputs.next().ok_or("--output requires a path")?.into());
                }
                Some(option) if option.starts_with('-') => {
                    return Err(
                        format!("unknown option {option} (use -- before such a filename)").into(),
                    );
                }
                _ => args.paths.push(arg.into()),
            }
        }
        if args.paths.is_empty() {
            return Err(USAGE.into());
        }
        Ok(Some(args))
    }
}

fn write_vectors(mut writer: impl Write, results: &[FileEmbedding]) -> io::Result<()> {
    for result in results {
        for value in &result.embedding {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()
}

fn save_vectors(
    output: &Path,
    inputs: &[PathBuf],
    results: &[FileEmbedding],
) -> Result<(), Box<dyn Error>> {
    let target = match fs::canonicalize(output) {
        Ok(path) => {
            for input in inputs {
                if same_file::is_same_file(&path, input)? {
                    return Err(format!("output is also an input: {}", output.display()).into());
                }
            }
            if !fs::metadata(&path)?.is_file() {
                return Ok(write_vectors(
                    BufWriter::new(File::options().write(true).open(path)?),
                    results,
                )?);
            }
            path
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => output.to_owned(),
        Err(err) => return Err(err.into()),
    };
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    write_vectors(BufWriter::new(temp.as_file_mut()), results)?;
    temp.persist(&target)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(args) = Args::parse()? else {
        println!("{USAGE}");
        return Ok(());
    };
    let model = NanoEncoder::baked()?;
    let start = Instant::now();
    let batch = encode_files(&model, &args.paths, args.mode, args.jobs);
    let elapsed = start.elapsed();
    let results: Vec<_> = batch
        .into_iter()
        .zip(&args.paths)
        .map(|(result, path)| result.map_err(|err| format!("{}: {err}", path.display())))
        .collect::<Result<_, _>>()?;

    if let Some(output) = &args.output {
        save_vectors(output, &args.paths, &results)?;
    } else {
        let mut writer = BufWriter::new(io::stdout().lock());
        for result in &results {
            write!(writer, "[")?;
            for (i, value) in result.embedding.iter().enumerate() {
                if i > 0 {
                    write!(writer, ",")?;
                }
                write!(writer, "{value}")?;
            }
            writeln!(writer, "]")?;
        }
        writer.flush()?;
    }
    if args.profile {
        eprintln!("read_ms\tdecode_ms\tresize_ms\tinference_ms\ttotal_ms\tpath");
        for (result, path) in results.iter().zip(&args.paths) {
            let t = result.timings;
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
                ms(t.read),
                ms(t.decode),
                ms(t.resize),
                ms(t.inference),
                ms(t.read + t.decode + t.resize + t.inference),
                path.display()
            );
        }
        eprintln!(
            "{} images in {:.3} ms ({:.1} images/s, excluding model load and output)",
            results.len(),
            elapsed.as_secs_f64() * 1000.0,
            results.len() as f64 / elapsed.as_secs_f64()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("nano_image: {err}");
            ExitCode::FAILURE
        }
    }
}
