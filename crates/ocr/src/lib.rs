//! pdfcraft-ocr — Scan & OCR ▸ Recognize text (L4).
//!
//! Recognition runs the ocrs engine (MIT/Apache-2.0) with its pre-trained models (CC-BY-SA-4.0,
//! fetched by `cargo xtask models`; see ATTRIBUTION.toml). The caller renders a page to pixels;
//! [`Ocr::recognize`] finds the words in them, and [`text_layer`] turns words placed in user space
//! into page content: invisible text (rendering mode 3) over each word, so the page becomes a
//! searchable image (Acrobat's "Searchable Image (Exact)": the image is left untouched).
//!
//! The models read the Latin alphabet (English and other languages written without accents).

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::path::{Path, PathBuf};

pub use pdfcraft_fonts::helvetica_width;

/// The model files, as named in ATTRIBUTION.toml.
pub const DETECTION_MODEL: &str = "text-detection.rten";
pub const RECOGNITION_MODEL: &str = "text-recognition.rten";

/// The recognition model's alphabet, as it was trained (ocrs-models' `DEFAULT_ALPHABET`): its
/// output classes follow this order. The ocrs crate's own copy has an `E` where the training had
/// `€`, which made a scanned "€250" read as "E250".
pub const ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// The languages the models read (ISO 639-1); all use the Latin alphabet without accents.
pub const LANGUAGES: &[(&str, &str)] = &[("en", "English")];

#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    /// Shown to end users (release packages ship the models; a source build fetches them with
    /// `cargo xtask models`). The UI catalogs key this exact text.
    #[error("the text recognition models are not installed (reinstall PdfTakari, or set PDFCRAFT_MODELS to the folder that holds them)")]
    NoModels,
    #[error("loading {0}: {1}")]
    Load(String, String),
    #[error("text recognition failed: {0}")]
    Recognize(String),
    #[error("the image is empty")]
    EmptyImage,
}

/// Where the two model files are.
#[derive(Clone, Debug, PartialEq)]
pub struct Models {
    pub detection: PathBuf,
    pub recognition: PathBuf,
}

impl Models {
    /// The models in `dir`, if both files are there.
    pub fn in_dir(dir: &Path) -> Option<Models> {
        let m = Models { detection: dir.join(DETECTION_MODEL), recognition: dir.join(RECOGNITION_MODEL) };
        (m.detection.is_file() && m.recognition.is_file()).then_some(m)
    }

    /// Look for the models: `$PDFCRAFT_MODELS`, then where the release packages install them
    /// beside the executable ([`Models::dirs_beside_exe`]), then the source tree's `assets/models/`.
    pub fn find() -> Option<Models> {
        Self::search_dirs().iter().find_map(|d| Self::in_dir(d))
    }

    /// The directories [`Models::find`] looks in, in order.
    pub fn search_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        if let Some(d) = std::env::var_os("PDFCRAFT_MODELS") {
            dirs.push(PathBuf::from(d));
        }
        if let Some(exe) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
            dirs.extend(Self::dirs_beside_exe(&exe));
        }
        dirs.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models"));
        dirs
    }

    /// Where the release packages put the models, relative to the directory holding the
    /// executable (packaging/*):
    /// - `models/` beside it: the Windows MSI and portable zip;
    /// - `../Resources/models`: the macOS app bundle (`Contents/MacOS` → `Contents/Resources`);
    /// - `../share/pdfcraft/models`: the FHS-style trees (`bin/` → `share/`) of the Linux
    ///   deb/rpm/tar.gz/AppImage/Flatpak and the FreeBSD tarball, wherever they are installed.
    pub fn dirs_beside_exe(exe_dir: &Path) -> Vec<PathBuf> {
        vec![exe_dir.join("models"), exe_dir.join("../Resources/models"), exe_dir.join("../share/pdfcraft/models")]
    }
}

/// A recognised word: its text, its box in image pixels `[left, top, right, bottom]`, and how
/// sure the recogniser is of it (0–1: the least likely of its letters; 1 when not measured).
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub rect: [f32; 4],
    pub confidence: f32,
}

/// A recognised line of words, in reading order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Line {
    pub words: Vec<Word>,
}

impl Line {
    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// A loaded recogniser. Loading takes a moment; keep one and reuse it.
pub struct Ocr {
    engine: ocrs::OcrEngine,
    /// A second copy of the recognition model, run again on each line for the letters'
    /// probabilities (ocrs keeps them to itself).
    recognizer: rten::Model,
}

impl Ocr {
    pub fn load(models: &Models) -> Result<Ocr, OcrError> {
        let load = |p: &Path| rten::Model::load_file(p).map_err(|e| OcrError::Load(p.display().to_string(), e.to_string()));
        let params = ocrs::OcrEngineParams {
            detection_model: Some(load(&models.detection)?),
            recognition_model: Some(load(&models.recognition)?),
            alphabet: Some(ALPHABET.into()),
            ..Default::default()
        };
        let engine = ocrs::OcrEngine::new(params).map_err(|e| OcrError::Load("ocr engine".into(), e.to_string()))?;
        Ok(Ocr { engine, recognizer: load(&models.recognition)? })
    }

    /// Load the models found by [`Models::find`].
    pub fn find() -> Result<Ocr, OcrError> {
        Self::load(&Models::find().ok_or(OcrError::NoModels)?)
    }

    /// Recognise the text in an RGBA (or RGB, or grey) image, `width` × `height` pixels.
    pub fn recognize(&self, pixels: &[u8], width: u32, height: u32) -> Result<Vec<Line>, OcrError> {
        if width == 0 || height == 0 || pixels.is_empty() {
            return Err(OcrError::EmptyImage);
        }
        // ocrs wants 1 or 3 channels.
        let rgb: std::borrow::Cow<[u8]> = if pixels.len() == (width * height * 4) as usize {
            pixels.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect::<Vec<u8>>().into()
        } else {
            pixels.into()
        };
        let err = |e: &dyn std::fmt::Display| OcrError::Recognize(e.to_string());
        let source = ocrs::ImageSource::from_bytes(&rgb, (width, height)).map_err(|e| err(&e))?;
        let input = self.engine.prepare_input(source).map_err(|e| err(&e))?;
        let found = self.engine.detect_words(&input).map_err(|e| err(&e))?;
        let lines = self.engine.find_text_lines(&input, &found);
        let read = self.engine.recognize_text(&input, &lines).map_err(|e| err(&e))?;
        use ocrs::TextItem;
        Ok(read
            .into_iter()
            .zip(&lines)
            .filter_map(|(line, rects)| {
                let line = line?;
                let mut words: Vec<Word> = line
                    .words()
                    .map(|w| {
                        let r = w.bounding_rect();
                        Word { text: w.to_string(), rect: [r.left() as f32, r.top() as f32, r.right() as f32, r.bottom() as f32], confidence: 1.0 }
                    })
                    .filter(|w| !w.text.trim().is_empty())
                    .collect();
                if let Some(read) = self.confidences(&input, rects) {
                    assign_confidences(&mut words, &read);
                }
                (!words.is_empty()).then_some(Line { words })
            })
            .collect())
    }

    /// Read one line again and return its words with the probability of their least likely
    /// letter. `None` when the model's output isn't the expected shape.
    fn confidences(&self, input: &ocrs::OcrInput, line: &[rten_imageproc::RotatedRect]) -> Option<Vec<(String, f32)>> {
        use rten_tensor::prelude::*;
        let image = self.engine.prepare_recognition_input(input, line).ok()?;
        let [h, w] = image.shape();
        let batch = rten_tensor::NdTensor::from_data([1, 1, h, w], image.to_vec());
        let out = self.recognizer.run_one(batch.view().as_dyn().into(), None).ok()?;
        let out: rten_tensor::NdTensor<f32, 3> = out.try_into().ok()?;
        // [sequence, batch, class]: log probabilities, class 0 the CTC blank.
        let [steps, _, classes] = out.shape();
        if classes != ALPHABET.chars().count() + 1 {
            return None;
        }
        let alphabet: Vec<char> = ALPHABET.chars().collect();
        let mut letters: Vec<(char, f32)> = Vec::new();
        let mut last = 0;
        for s in 0..steps {
            let (mut best, mut p) = (0, f32::NEG_INFINITY);
            for c in 0..classes {
                let v = out.get([s, 0, c]).copied().unwrap_or(f32::NEG_INFINITY);
                if v > p {
                    (best, p) = (c, v);
                }
            }
            let p = if p.is_finite() { p.exp().clamp(0.0, 1.0) } else { 0.0 };
            if best != last && best > 0 {
                letters.push((alphabet.get(best - 1).copied().unwrap_or('?'), p));
            } else if best == last
                && best > 0
                && let Some(l) = letters.last_mut()
            {
                l.1 = l.1.max(p);
            }
            last = best;
        }
        let mut words = Vec::new();
        let mut cur: Option<(String, f32)> = None;
        for (c, p) in letters {
            if c == ' ' {
                words.extend(cur.take());
                continue;
            }
            let w = cur.get_or_insert_with(|| (String::new(), 1.0));
            w.0.push(c);
            w.1 = w.1.min(p);
        }
        words.extend(cur);
        Some(words)
    }
}

/// Give each word the confidence of the same word read again: in order when the two readings
/// split the line into as many words, else by matching text; a word not found takes the
/// line's lowest.
fn assign_confidences(words: &mut [Word], read: &[(String, f32)]) {
    if read.len() == words.len() {
        for (w, r) in words.iter_mut().zip(read) {
            w.confidence = r.1;
        }
        return;
    }
    let lowest = read.iter().map(|r| r.1).fold(1.0f32, f32::min);
    let mut from = 0;
    for w in words.iter_mut() {
        match read.get(from..).and_then(|rest| rest.iter().position(|r| r.0 == w.text)) {
            Some(i) => {
                w.confidence = read.get(from + i).map_or(lowest, |r| r.1);
                from += i + 1;
            }
            None => w.confidence = lowest,
        }
    }
}

/// A word placed on a page, in PDF user space: the bottom-left corner of its box, and the
/// vectors along its bottom edge (`across`, left → right as read) and its left edge (`up`).
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedWord {
    pub text: String,
    pub origin: [f64; 2],
    pub across: [f64; 2],
    pub up: [f64; 2],
}

impl PlacedWord {
    /// Place a pixel-space word with `to_user`, which maps an image pixel (x right, y down) to
    /// user space.
    pub fn place(word: &Word, to_user: impl Fn(f32, f32) -> [f64; 2]) -> PlacedWord {
        let [l, t, r, b] = word.rect;
        let o = to_user(l, b);
        let br = to_user(r, b);
        let tl = to_user(l, t);
        PlacedWord { text: word.text.clone(), origin: o, across: [br[0] - o[0], br[1] - o[1]], up: [tl[0] - o[0], tl[1] - o[1]] }
    }
}

/// Helvetica's ascender and descender (em fractions): a word box spans about this much.
const BOX_EM: f64 = 0.93;
const DESCENT: f64 = 0.21;

fn num(v: f64) -> String {
    let s = format!("{:.4}", if v.abs() < 1e-9 { 0.0 } else { v });
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}

/// Page content that writes `words` as invisible text in `/PCHelv` (standard Helvetica,
/// WinAnsiEncoding), each word stretched to its box so selection and search highlight the
/// right place. Marked content `/OCR` so it can be told apart from the page's own text.
pub fn text_layer(words: &[PlacedWord]) -> Vec<u8> {
    let mut out = b"/OCR BMC\nBT\n3 Tr\n/PCHelv 1 Tf\n".to_vec();
    for w in words {
        let width = helvetica_width(&w.text, 1.0);
        let height = w.up[0].hypot(w.up[1]);
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        // The em square: its height spans the box; its width is stretched to fit the word.
        let em = [w.up[0] / BOX_EM, w.up[1] / BOX_EM];
        let ax = [w.across[0] / width, w.across[1] / width];
        let o = [w.origin[0] + em[0] * DESCENT, w.origin[1] + em[1] * DESCENT];
        let m = [ax[0], ax[1], em[0], em[1], o[0], o[1]];
        let nums: Vec<String> = m.iter().map(|v| num(*v)).collect();
        out.extend_from_slice(nums.join(" ").as_bytes());
        out.extend_from_slice(b" Tm ");
        out.extend(pdfcraft_fonts::literal(&pdfcraft_fonts::win_ansi(&w.text)));
        out.extend_from_slice(b" Tj\n");
    }
    out.extend_from_slice(b"ET\nEMC\n");
    out
}

/// A recognised line placed on a page, with the colours of its letters (`ink`) and of the paper
/// around them (`paper`), as RGB in 0..=1, sampled from the scan by [`line_colours`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedLine {
    pub words: Vec<PlacedWord>,
    pub ink: [f64; 3],
    pub paper: [f64; 3],
    /// The letters' em box measured from the scan ([`ink_box`], [`em_box`]): `origin` on the
    /// baseline where the ink starts, `across` along the baseline to where it ends, `up` one em.
    /// `None`: estimated from the word boxes, which OCR pads unevenly.
    pub em: Option<PlacedWord>,
}

impl PlacedLine {
    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// At most this many pixels are sampled per region, so a huge box costs no more than a small one.
const MAX_SAMPLES: usize = 4096;

/// The colours of the letters and of the paper in the box `rect` (`[left, top, right, bottom]` in
/// pixels) of an RGBA image: the paper is the median of a ring just outside the box, the ink the
/// median of the box's pixels that differ most from it. A faint or unreadable box gets black ink
/// on white (or white on a dark paper), so the text stays legible.
pub fn line_colours(rgba: &[u8], width: u32, height: u32, rect: [f32; 4]) -> ([f64; 3], [f64; 3]) {
    const BLACK: [f64; 3] = [0.0, 0.0, 0.0];
    const WHITE: [f64; 3] = [1.0, 1.0, 1.0];
    let (w, h) = (width as usize, height as usize);
    if w.checked_mul(h).and_then(|n| n.checked_mul(4)).is_none_or(|n| rgba.len() < n) || w == 0 || h == 0 {
        return (BLACK, WHITE);
    }
    let clamp = |v: f32, max: usize| if v.is_finite() { (v.max(0.0) as usize).min(max) } else { 0 };
    let [l, t] = [clamp(rect[0].floor(), w), clamp(rect[1].floor(), h)];
    let [r, b] = [clamp(rect[2].ceil(), w), clamp(rect[3].ceil(), h)];
    if r <= l || b <= t {
        return (BLACK, WHITE);
    }
    let pad = ((b - t) / 5).max(2);
    let outer = [l.saturating_sub(pad), t.saturating_sub(pad), r.saturating_add(pad).min(w), b.saturating_add(pad).min(h)];
    let pixel = |x: usize, y: usize| -> Option<[u8; 3]> {
        let o = y.checked_mul(w)?.checked_add(x)?.checked_mul(4)?;
        <[u8; 3]>::try_from(rgba.get(o..o.checked_add(3)?)?).ok()
    };
    // Every `step`th pixel each way, so a region yields about MAX_SAMPLES at most.
    let samples = |[x0, y0, x1, y1]: [usize; 4], keep: &dyn Fn(usize, usize) -> bool| -> Vec<[u8; 3]> {
        let area = (x1 - x0).saturating_mul(y1 - y0);
        let step = ((area / MAX_SAMPLES) as f64).sqrt().ceil().max(1.0) as usize;
        (y0..y1)
            .step_by(step)
            .flat_map(|y| (x0..x1).step_by(step).map(move |x| (x, y)))
            .filter(|&(x, y)| keep(x, y))
            .filter_map(|(x, y)| pixel(x, y))
            .collect()
    };
    let inside = samples([l, t, r, b], &|_, _| true);
    let ring = samples(outer, &|x, y| x < l || x >= r || y < t || y >= b);
    let median = |px: &[[u8; 3]]| -> Option<[f64; 3]> {
        let mut out = [0.0; 3];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut v: Vec<u8> = px.iter().map(|p| p[c]).collect();
            v.sort_unstable();
            *slot = f64::from(*v.get(v.len() / 2)?) / 255.0;
        }
        Some(out)
    };
    // A box filling the whole image has no ring; most of a text box is paper anyway.
    let Some(paper) = median(&ring).or_else(|| median(&inside)) else { return (BLACK, WHITE) };
    let distance = |p: &[u8; 3]| (0..3).map(|c| (f64::from(p[c]) / 255.0 - paper[c]).abs()).sum::<f64>();
    let mut by_contrast = inside;
    by_contrast.sort_unstable_by(|a, b| distance(b).total_cmp(&distance(a)));
    // The strongest few: a slightly blurred scan greys the edges of every stroke.
    by_contrast.truncate((by_contrast.len() / 20).max(1));
    let ink = median(&by_contrast).filter(|i| (0..3).map(|c| (i[c] - paper[c]).abs()).sum::<f64>() > 0.25);
    let light = 0.299 * paper[0] + 0.587 * paper[1] + 0.114 * paper[2] > 0.5;
    (ink.unwrap_or(if light { BLACK } else { WHITE }), paper)
}

/// Boxes larger than this many pixels aren't measured ([`ink_box`]), so a misdetected
/// page-sized line costs nothing.
const MAX_MEASURED: usize = 4_000_000;

/// The tight box (`[left, top, right, bottom]`, pixels) of the letters inside `rect`: the pixels
/// nearer `ink` than `paper`. OCR word boxes are padded unevenly, and may reach into the lines
/// above and below; the line is the run of ink rows around its densest row, across gaps of at
/// most an eighth of the box (the dot of an i stays, a neighbour's descenders don't).
pub fn ink_box(rgba: &[u8], width: u32, height: u32, rect: [f32; 4], ink: [f64; 3], paper: [f64; 3]) -> Option<[f32; 4]> {
    let (w, h) = (width as usize, height as usize);
    if w.checked_mul(h)?.checked_mul(4)? > rgba.len() {
        return None;
    }
    let clamp = |v: f32, max: usize| if v.is_finite() { (v.max(0.0) as usize).min(max) } else { 0 };
    let [l, t] = [clamp(rect[0].floor(), w), clamp(rect[1].floor(), h)];
    let [r, b] = [clamp(rect[2].ceil(), w), clamp(rect[3].ceil(), h)];
    if r <= l || b <= t || (r - l).saturating_mul(b - t) > MAX_MEASURED {
        return None;
    }
    let distance = |p: &[u8], c: [f64; 3]| (0..3).map(|i| (f64::from(p.get(i).copied().unwrap_or(0)) / 255.0 - c[i]).abs()).sum::<f64>();
    let is_ink = |x: usize, y: usize| {
        // x < w and y < h, and w × h × 4 fits the buffer (checked above): no overflow.
        let o = (y * w + x) * 4;
        rgba.get(o..o + 3).is_some_and(|p| {
            let (to_paper, to_ink) = (distance(p, paper), distance(p, ink));
            to_paper > 0.25 && to_ink < to_paper
        })
    };
    let rows: Vec<usize> = (t..b).map(|y| (l..r).filter(|&x| is_ink(x, y)).count()).collect();
    let densest = rows.iter().enumerate().max_by_key(|(_, n)| **n).filter(|(_, n)| **n > 0)?.0;
    let gap = (rows.len() / 8).max(1);
    let extend = |range: &mut dyn Iterator<Item = usize>| {
        let (mut edge, mut blank) = (densest, 0);
        for i in range {
            if rows.get(i).is_some_and(|n| *n > 0) {
                edge = i;
                blank = 0;
            } else {
                blank += 1;
                if blank > gap {
                    break;
                }
            }
        }
        edge
    };
    let top = extend(&mut (0..densest).rev());
    let bottom = extend(&mut (densest + 1..rows.len()));
    let cols = (l..r).filter(|&x| (t + top..=t + bottom).any(|y| is_ink(x, y)));
    let (left, right) = cols.fold((usize::MAX, 0), |(a, z), x| (a.min(x), z.max(x)));
    (left <= right).then(|| [left as f32, (t + top) as f32, right as f32 + 1.0, (t + bottom) as f32 + 1.0])
}

/// Helvetica's cap height and x-height (em fractions), with [`DESCENT`] what a line's ink spans.
const CAP_HEIGHT: f64 = 0.718;
const X_HEIGHT: f64 = 0.523;

/// The em box of the letters of `text` whose ink fills `tight` (pixels, from [`ink_box`]): a
/// [`Word`] whose box runs from the baseline (its bottom) up one em. The ink reaches the cap
/// height when the text has capitals, digits or ascenders (else the x-height), and the descent
/// when it has descenders (else it sits on the baseline).
pub fn em_box(tight: [f32; 4], text: &str) -> Option<Word> {
    let tall = text.chars().any(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || "bdfhklt!?/\\'\"()[]{}|#%&@$€".contains(c));
    let deep = text.chars().any(|c| "gjpqy,;()[]{}|".contains(c));
    let span = if tall { CAP_HEIGHT } else { X_HEIGHT } + if deep { DESCENT } else { 0.0 };
    let em = f64::from(tight[3] - tight[1]) / span;
    let baseline = f64::from(tight[3]) - if deep { DESCENT * em } else { 0.0 };
    if !(em.is_finite() && em > 0.0 && tight[2] > tight[0]) {
        return None;
    }
    Some(Word { text: text.into(), rect: [tight[0], (baseline - em) as f32, tight[2], baseline as f32], confidence: 1.0 })
}

/// A recognised line ready for Editable Text output, in image pixels: its words, the colours of
/// its letters and paper ([`line_colours`]) and its measured em box ([`ink_box`], [`em_box`]).
#[derive(Clone, Debug, PartialEq)]
pub struct EditableLine {
    pub words: Vec<Word>,
    pub ink: [f64; 3],
    pub paper: [f64; 3],
    pub em: Option<Word>,
}

/// A word this many times taller than the others on its line has handwriting or a stamp through
/// it (a signature's loops stretch the box OCR finds), so it is left in the image.
const TALL_WORD: f32 = 1.8;
/// A word shorter than this fraction of the page's typical word is a speck, not a letter.
const SPECK: f32 = 0.4;
/// A word nearer than this many line heights to one left in the image (on any line, side by
/// side) is left too: a stamp or signature rarely ends exactly where OCR split the words.
const TOUCHING: f32 = 0.5;
/// A word the recogniser is less sure of than this is left in the image: replacing scanned words
/// with a misreading ("Area L?de" for "Area Code") loses what the page said, while leaving a
/// correct word only means it can't be edited. Clean print reads at 0.95 and above.
const UNSURE: f32 = 0.9;
/// Words further apart than this many line heights are separate fields (columns, form labels
/// and their values), each its own line.
const FIELD_GAP: f32 = 2.5;

/// Prepare the lines OCR found for "Editable Text and Images": only printed words become text.
/// - Words much taller than the rest of their line (handwriting or a stamp through them), in an
///   ink of another colour than the page's print (a blue signature, a red stamp) or too small to
///   be letters (specks) are left in the image, untouched.
/// - Words touching those are left too ([`TOUCHING`]), and so are words the recogniser isn't
///   sure of ([`UNSURE`]), so a misreading never replaces what the scan says.
/// - A line is split where such a word was, at a ruled line between two words (table cells), at
///   a gap wider than [`FIELD_GAP`] line heights, so each cell or field edits on its own, and
///   where something OCR didn't read is inked between two words, so the cover never hides it.
/// - A measured em that doesn't fit the words' boxes (something else inked in the box) is
///   dropped, and the size is estimated from the boxes instead.
pub fn editable_lines(rgba: &[u8], width: u32, height: u32, lines: &[Line]) -> Vec<EditableLine> {
    let tall = |w: &Word| w.rect[3] - w.rect[1];
    let sane = |w: &&Word| w.rect.iter().all(|v| v.is_finite()) && tall(w) > 0.0 && w.rect[2] > w.rect[0];
    let page_height = lower_median(lines.iter().flat_map(|l| &l.words).filter(sane).map(tall).collect());
    // Each word's colours, and the page's print colour: the median ink of its words.
    let coloured: Vec<Vec<Coloured>> = lines
        .iter()
        .map(|l| {
            l.words
                .iter()
                .filter(sane)
                .map(|w| {
                    let (ink, paper) = line_colours(rgba, width, height, w.rect);
                    (w, ink, paper)
                })
                .collect()
        })
        .collect();
    let inks: Vec<[f64; 3]> = coloured.iter().flatten().map(|(_, ink, _)| *ink).collect();
    let print = std::array::from_fn::<f64, 3, _>(|c| lower_median(inks.iter().map(|i| i[c] as f32).collect()).map_or(0.0, f64::from));
    let chroma = |c: [f64; 3]| c.iter().copied().fold(f64::MIN, f64::max) - c.iter().copied().fold(f64::MAX, f64::min);
    // A faded stamp is barely tinted once blurred; a printed word mistaken for one only stays in
    // the image, so this errs that way.
    let foreign = |ink: [f64; 3]| chroma(ink) - chroma(print) > 0.025 && (0..3).map(|c| (ink[c] - print[c]).abs()).sum::<f64>() > 0.12;
    // The marks (handwriting, stamps, specks) across the page, each with how near a word must
    // come to it to be left in the image with it: a stamp's date can sit on another OCR line.
    let mut marks: Vec<([f32; 4], f32)> = Vec::new();
    let mut heights = Vec::new();
    for words in &coloured {
        let line_height = lower_median(words.iter().map(|(w, ..)| tall(w)).collect());
        heights.push(line_height);
        let Some(line_height) = line_height else { continue };
        for &(w, ink, _) in words {
            let print_like = tall(w) <= TALL_WORD * line_height && page_height.is_none_or(|p| tall(w) >= SPECK * p) && !foreign(ink);
            if !print_like {
                marks.push((w.rect, TOUCHING * line_height));
            }
        }
    }
    let marked =
        |w: &Word| marks.iter().any(|&(m, reach)| w.rect[1] < m[3] && m[1] < w.rect[3] && w.rect[0] - m[2] < reach && m[0] - w.rect[2] < reach);
    let mut out = Vec::new();
    for (words, line_height) in coloured.iter().zip(heights) {
        let Some(line_height) = line_height else { continue };
        let mut part: Vec<Word> = Vec::new();
        for &(w, _, paper) in words {
            if marked(w) || w.confidence < UNSURE {
                out.extend(measure(rgba, width, height, std::mem::take(&mut part)));
                continue;
            }
            if let Some(prev) = part.last()
                && splits(rgba, width, height, prev.rect, w.rect, line_height, paper)
            {
                out.extend(measure(rgba, width, height, std::mem::take(&mut part)));
            }
            part.push(w.clone());
        }
        out.extend(measure(rgba, width, height, part));
    }
    out
}

/// Whether the line breaks between neighbouring words `a` and `b` (pixel boxes, left to right):
/// they are separate fields ([`FIELD_GAP`]), a ruled line runs between them (a table's cells), or
/// something OCR didn't read is inked in a wide gap between them (a digit, a symbol), which a
/// cover across the whole line would hide.
fn splits(rgba: &[u8], width: u32, height: u32, a: [f32; 4], b: [f32; 4], line_height: f32, paper: [f64; 3]) -> bool {
    let gap = b[0] - a[2];
    if gap > FIELD_GAP * line_height {
        return true;
    }
    // A cell's border runs past the letters above and below; a letter's stem clipped by its word
    // box doesn't.
    let reach = 0.25 * line_height;
    let (top, bottom) = (a[1].min(b[1]), a[3].max(b[3]));
    let ruled = ink_columns(rgba, width, height, [a[2] + 1.0, top - reach, b[0] - 1.0, bottom + reach], paper)
        .is_some_and(|(cols, rows)| cols.iter().any(|&n| 20 * n >= 17 * rows));
    // An ordinary space between words holds at most the edges of letters OCR's boxes clipped;
    // only a wider gap, away from its edges, is looked in.
    let margin = 0.2 * line_height;
    let unread = gap > 0.6 * line_height
        && ink_columns(rgba, width, height, [a[2] + margin, top, b[0] - margin, bottom], paper).is_some_and(|(cols, rows)| {
            let inked: usize = cols.iter().sum();
            inked >= 4 && 50 * inked >= cols.len().saturating_mul(rows)
        });
    ruled || unread
}

/// The number of inked pixels (far from `paper`) in each column of `rect` (`[left, top, right,
/// bottom]`, pixels), and the number of rows; `None` for a box outside the image, of fewer than
/// three rows, or larger than [`MAX_MEASURED`].
fn ink_columns(rgba: &[u8], width: u32, height: u32, rect: [f32; 4], paper: [f64; 3]) -> Option<(Vec<usize>, usize)> {
    let (w, h) = (width as usize, height as usize);
    if w.checked_mul(h)?.checked_mul(4)? > rgba.len() {
        return None;
    }
    let clamp = |v: f32, max: usize| if v.is_finite() { (v.max(0.0) as usize).min(max) } else { 0 };
    let [l, t] = [clamp(rect[0].ceil(), w), clamp(rect[1].floor(), h)];
    let [r, b] = [clamp(rect[2].floor(), w), clamp(rect[3].ceil(), h)];
    if r <= l || b < t.saturating_add(3) || (r - l).saturating_mul(b - t) > MAX_MEASURED {
        return None;
    }
    let inked = |x: usize, y: usize| {
        // x < w and y < h, and w × h × 4 fits the buffer (checked above): no overflow.
        let o = (y * w + x) * 4;
        rgba.get(o..o + 3).is_some_and(|p| p.iter().zip(paper).map(|(v, c)| (f64::from(*v) / 255.0 - c).abs()).sum::<f64>() > 0.35)
    };
    Some(((l..r).map(|x| (t..b).filter(|&y| inked(x, y)).count()).collect(), b - t))
}

/// A word with the colours of its letters and of the paper around it ([`line_colours`]).
type Coloured<'a> = (&'a Word, [f64; 3], [f64; 3]);

/// The lower median (the smaller middle value of an even count), ignoring NaNs.
fn lower_median(mut v: Vec<f32>) -> Option<f32> {
    v.retain(|x| x.is_finite());
    v.sort_unstable_by(f32::total_cmp);
    v.get(v.len().checked_sub(1)? / 2).copied()
}

/// An [`EditableLine`] of `words` (none if empty): its colours and its em, kept only when it is
/// within reach of what the word boxes suggest.
fn measure(rgba: &[u8], width: u32, height: u32, words: Vec<Word>) -> Option<EditableLine> {
    let rect = words.iter().map(|w| w.rect).reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])])?;
    let text = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    let (ink, paper) = line_colours(rgba, width, height, rect);
    let expected = lower_median(words.iter().map(|w| w.rect[3] - w.rect[1]).collect())? / BOX_EM as f32;
    let em = ink_box(rgba, width, height, rect, ink, paper)
        .and_then(|tight| em_box(tight, &text))
        .filter(|e| (0.6..=1.6).contains(&((e.rect[3] - e.rect[1]) / expected)));
    Some(EditableLine { words, ink, paper, em })
}

/// Horizontal scaling (`Tz`, percent) is kept within this range, so a misread line doesn't come
/// out squashed to nothing or stretched across the page.
const SCALE_RANGE: (f64, f64) = (50.0, 200.0);

/// Page content for Acrobat's "Editable Text and Images" output: each line's scanned words are
/// covered with the paper colour, then the line is drawn as visible text in `/PCHelv` (standard
/// Helvetica, WinAnsiEncoding) in the ink colour, sized to the line's height and scaled to its
/// width. Each line is one text object, so Edit text sees it as one editable line. The covers all
/// come first, so one line's margin never paints over another line's text.
pub fn editable_layer(lines: &[PlacedLine]) -> Vec<u8> {
    let mut covers = String::new();
    let mut texts = Vec::new();
    for line in lines {
        let Some(g) = LineGeometry::of(line) else { continue };
        let text = line.text();
        if text.trim().is_empty() {
            continue;
        }
        // Size, baseline and width from the measured letters, else estimated from the boxes.
        let (size, origin, across, up, length) = match line.em.as_ref().and_then(Measured::of) {
            Some(m) => (m.em, m.origin, m.across, m.up, m.length),
            None => {
                let size = g.height / BOX_EM;
                let up = DESCENT * size;
                (size, [g.base[0] + g.up[0] * up, g.base[1] + g.up[1] * up], g.across, g.up, g.length)
            }
        };
        let natural = helvetica_width(&text, size);
        if natural <= 0.0 {
            continue;
        }
        // The cover: the line's box, a little larger, so no edge of a letter shows.
        let pad = g.height * 0.12;
        let (a, u) = (g.across, g.up);
        let at = |along: f64, up: f64| [g.base[0] + a[0] * along + u[0] * up, g.base[1] + a[1] * along + u[1] * up];
        let corners = [at(-pad, -pad), at(g.length + pad, -pad), at(g.length + pad, g.height + pad), at(-pad, g.height + pad)];
        let [r, gr, b] = line.paper;
        covers.push_str(&format!("{} {} {} rg\n", num(r), num(gr), num(b)));
        for (i, c) in corners.iter().enumerate() {
            covers.push_str(&format!("{} {} {}\n", num(c[0]), num(c[1]), if i == 0 { "m" } else { "l" }));
        }
        covers.push_str("h f\n");
        // The text, on the baseline and scaled to the line's width.
        let scale = (100.0 * length / natural).clamp(SCALE_RANGE.0, SCALE_RANGE.1);
        let [r, gr, b] = line.ink;
        let m = [across[0], across[1], up[0], up[1], origin[0], origin[1]].map(num).join(" ");
        texts.extend_from_slice(format!("BT\n/PCHelv {} Tf\n{} Tz\n{} {} {} rg\n{m} Tm ", num(size), num(scale), num(r), num(gr), num(b)).as_bytes());
        texts.extend(pdfcraft_fonts::literal(&pdfcraft_fonts::win_ansi(&text)));
        texts.extend_from_slice(b" Tj\nET\n");
    }
    let mut out = b"q\n".to_vec();
    out.extend_from_slice(covers.as_bytes());
    out.extend(texts);
    out.extend_from_slice(b"Q\n");
    out
}

/// A measured em box ([`PlacedLine::em`]) as unit vectors, the em and the baseline's length.
struct Measured {
    origin: [f64; 2],
    across: [f64; 2],
    up: [f64; 2],
    em: f64,
    length: f64,
}

impl Measured {
    fn of(w: &PlacedWord) -> Option<Measured> {
        let length = w.across[0].hypot(w.across[1]);
        let em = w.up[0].hypot(w.up[1]);
        if !(length.is_finite() && length > 0.0 && em.is_finite() && em > 0.0 && w.origin.iter().all(|v| v.is_finite())) {
            return None;
        }
        let across = [w.across[0] / length, w.across[1] / length];
        // Upright text, on the side the em rises to.
        let mut up = [-across[1], across[0]];
        if up[0] * w.up[0] + up[1] * w.up[1] < 0.0 {
            up = [-up[0], -up[1]];
        }
        Some(Measured { origin: w.origin, across, up, em, length })
    }
}

/// Where a line sits: its bottom-left corner, unit vectors along it and up from it, its length
/// and height (user space).
struct LineGeometry {
    base: [f64; 2],
    across: [f64; 2],
    up: [f64; 2],
    length: f64,
    height: f64,
}

impl LineGeometry {
    fn of(line: &PlacedLine) -> Option<LineGeometry> {
        let first = line.words.first()?;
        // The direction the words run: their baselines summed (a line's word bottoms differ with
        // descenders, so joining the first word's bottom to the last's would tilt it).
        let sum = line.words.iter().fold([0.0, 0.0], |s, w| [s[0] + w.across[0], s[1] + w.across[1]]);
        let norm = sum[0].hypot(sum[1]);
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        let across = [sum[0] / norm, sum[1] / norm];
        // Upright text: perpendicular to the line, on the side the words' boxes rise to.
        let mut up = [-across[1], across[0]];
        if up[0] * first.up[0] + up[1] * first.up[1] < 0.0 {
            up = [-up[0], -up[1]];
        }
        // The line's extent along and up from the first word's corner: from the lowest word
        // bottom (descenders) to the highest top (capitals), the first start to the last end.
        let dot = |v: [f64; 2], d: [f64; 2]| v[0] * d[0] + v[1] * d[1];
        let (mut left, mut right, mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for w in &line.words {
            let o = [w.origin[0] - first.origin[0], w.origin[1] - first.origin[1]];
            let (x, y) = (dot(o, across), dot(o, up));
            let (dx, dy) = (dot(w.across, across) + dot(w.up, across), dot(w.up, up) + dot(w.across, up));
            left = left.min(x).min(x + dx);
            right = right.max(x).max(x + dx);
            low = low.min(y).min(y + dy);
            high = high.max(y).max(y + dy);
        }
        let (length, height) = (right - left, high - low);
        if !(length.is_finite() && length > 0.0 && height.is_finite() && height > 0.0) {
            return None;
        }
        let base = [first.origin[0] + across[0] * left + up[0] * low, first.origin[1] + across[1] * left + up[1] * low];
        Some(LineGeometry { base, across, up, length, height })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_words_through_the_mapping() {
        // 2 pixels per point, page 612 × 792, y flipped.
        let to_user = |x: f32, y: f32| [x as f64 / 2.0, 792.0 - y as f64 / 2.0];
        let w = PlacedWord::place(&Word { text: "Hello".into(), rect: [100.0, 200.0, 300.0, 240.0], confidence: 1.0 }, to_user);
        assert_eq!(w.origin, [50.0, 672.0]);
        assert_eq!(w.across, [100.0, 0.0]);
        assert_eq!(w.up, [0.0, 20.0]);
    }

    #[test]
    fn text_layer_is_invisible_text_fitted_to_each_box() {
        let w = PlacedWord { text: "Hi".into(), origin: [10.0, 20.0], across: [30.0, 0.0], up: [0.0, 9.3] };
        let s = String::from_utf8(text_layer(&[w])).unwrap();
        assert!(s.contains("3 Tr") && s.contains("/PCHelv 1 Tf"), "{s}");
        let width = helvetica_width("Hi", 1.0);
        assert!(s.contains(&format!("{} 0 0 10 10 22.1 Tm (Hi) Tj", num(30.0 / width))), "{s}");
        assert!(s.starts_with("/OCR BMC") && s.ends_with("EMC\n"));
    }

    #[test]
    fn rotated_pages_keep_the_reading_direction() {
        // A page turned 90°: image x runs up the page.
        let to_user = |x: f32, y: f32| [y as f64, x as f64];
        let w = PlacedWord::place(&Word { text: "Up".into(), rect: [0.0, 0.0, 50.0, 10.0], confidence: 1.0 }, to_user);
        assert_eq!(w.across, [0.0, 50.0]);
        assert_eq!(w.up, [-10.0, 0.0]);
    }

    /// A scan's line: dark blue letters on cream paper, a white page elsewhere.
    fn tinted_line() -> (u32, u32, Vec<u8>) {
        let (w, h) = (120u32, 60u32);
        let mut px = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let colour = if (20..100).contains(&x) && (20..40).contains(&y) && x % 4 == 0 {
                    [20, 30, 120]
                } else if (10..110).contains(&x) && (12..48).contains(&y) {
                    [250, 240, 210]
                } else {
                    [255, 255, 255]
                };
                px[o..o + 3].copy_from_slice(&colour);
            }
        }
        (w, h, px)
    }

    #[test]
    fn line_colours_sample_ink_and_paper() {
        let (w, h, px) = tinted_line();
        let (ink, paper) = line_colours(&px, w, h, [20.0, 20.0, 100.0, 40.0]);
        let close = |a: [f64; 3], b: [u8; 3]| (0..3).all(|c| (a[c] - f64::from(b[c]) / 255.0).abs() < 0.02);
        assert!(close(ink, [20, 30, 120]), "{ink:?}");
        assert!(close(paper, [250, 240, 210]), "{paper:?}");
    }

    /// Never-panic: boxes outside the image, NaN, empty and short buffers all give a legible default.
    #[test]
    fn line_colours_survive_bad_input() {
        let (w, h, px) = tinted_line();
        let default = ([0.0; 3], [1.0; 3]);
        assert_eq!(line_colours(&px, w, h, [500.0, 500.0, 600.0, 600.0]), default);
        assert_eq!(line_colours(&px, w, h, [f32::NAN, 0.0, f32::INFINITY, 10.0]), default);
        assert_eq!(line_colours(&px[..10], w, h, [0.0, 0.0, 10.0, 10.0]), default);
        assert_eq!(line_colours(&[], 0, 0, [0.0, 0.0, 10.0, 10.0]), default);
        assert_eq!(line_colours(&px, u32::MAX, u32::MAX, [0.0, 0.0, 10.0, 10.0]), default);
        // A box filling the image has no ring: the paper comes from inside it.
        let flat = vec![200u8; 16 * 16 * 4];
        let (ink, paper) = line_colours(&flat, 16, 16, [0.0, 0.0, 16.0, 16.0]);
        assert!(paper.iter().all(|c| (c - 200.0 / 255.0).abs() < 1e-9), "{paper:?}");
        assert_eq!(ink, [0.0; 3], "no letters: black ink on the light paper");
    }

    /// White, with black blocks at `[left, top, right, bottom]`.
    fn blocks(w: u32, h: u32, rects: &[[u32; 4]]) -> Vec<u8> {
        let mut px = vec![255u8; (w * h * 4) as usize];
        for &[l, t, r, b] in rects {
            for y in t..b {
                for x in l..r {
                    let o = ((y * w + x) * 4) as usize;
                    px[o..o + 3].copy_from_slice(&[0, 0, 0]);
                }
            }
        }
        px
    }

    #[test]
    fn ink_box_finds_the_letters_not_the_padding_or_the_neighbours() {
        let (black, white) = ([0.0; 3], [1.0; 3]);
        // A line of "letters" at rows 40..60, an i's dot two rows above, and the line below's
        // ascenders reaching into the bottom of the padded OCR box.
        let px = blocks(200, 120, &[[30, 40, 150, 60], [60, 36, 64, 38], [40, 86, 140, 100]]);
        let tight = ink_box(&px, 200, 120, [20.0, 20.0, 170.0, 90.0], black, white);
        assert_eq!(tight, Some([30.0, 36.0, 150.0, 60.0]));
        // Nothing but paper; a box outside the image; a short buffer.
        assert_eq!(ink_box(&px, 200, 120, [160.0, 0.0, 200.0, 30.0], black, white), None);
        assert_eq!(ink_box(&px, 200, 120, [300.0, 300.0, 400.0, 400.0], black, white), None);
        assert_eq!(ink_box(&px[..100], 200, 120, [0.0, 0.0, 10.0, 10.0], black, white), None);
    }

    #[test]
    fn em_box_follows_capitals_and_descenders() {
        let tight = [10.0, 100.0, 210.0, 100.0 + 18.5];
        // Capitals and descenders: the ink spans cap height + descent.
        let w = em_box(tight, "Delivery").unwrap();
        let em = 18.5 / (CAP_HEIGHT + DESCENT);
        assert!(((w.rect[3] - w.rect[1]) as f64 - em).abs() < 1e-3, "{w:?}");
        assert!((w.rect[3] as f64 - (118.5 - DESCENT * em)).abs() < 1e-3, "baseline above the descenders: {w:?}");
        // Capitals only: the ink sits on the baseline and spans the cap height.
        let w = em_box(tight, "STATION 14").unwrap();
        assert!(((w.rect[3] - w.rect[1]) as f64 - 18.5 / CAP_HEIGHT).abs() < 1e-3 && w.rect[3] == 118.5, "{w:?}");
        // Only x-height letters.
        let w = em_box(tight, "a an once").unwrap();
        assert!(((w.rect[3] - w.rect[1]) as f64 - 18.5 / X_HEIGHT).abs() < 1e-3, "{w:?}");
        assert_eq!([w.rect[0], w.rect[2]], [10.0, 210.0]);
        assert_eq!(em_box([0.0, 10.0, 0.0, 20.0], "x"), None);
        assert_eq!(em_box([0.0, 10.0, 5.0, 10.0], "x"), None);
    }

    #[test]
    fn editable_layer_uses_the_measured_em() {
        let em = PlacedWord { text: String::new(), origin: [100.0, 503.0], across: [80.0, 0.0], up: [0.0, 12.0] };
        let line = PlacedLine { words: vec![word("Hello", 98.0, 44.0, 498.0, 20.0)], ink: [0.0; 3], paper: [1.0; 3], em: Some(em) };
        let s = String::from_utf8(editable_layer(&[line])).unwrap();
        // The measured size and baseline, not the padded box's.
        assert!(s.contains("/PCHelv 12 Tf") && s.contains("1 0 0 1 100 503 Tm"), "{s}");
        let natural = helvetica_width("Hello", 12.0);
        assert!(s.contains(&format!("{} Tz", num((100.0 * 80.0 / natural).clamp(50.0, 200.0)))), "{s}");
    }

    fn word(text: &str, x: f64, width: f64, bottom: f64, height: f64) -> PlacedWord {
        PlacedWord { text: text.into(), origin: [x, bottom], across: [width, 0.0], up: [0.0, height] }
    }

    #[test]
    fn editable_layer_covers_the_words_and_draws_visible_text() {
        let line = PlacedLine {
            words: vec![word("Hello", 100.0, 40.0, 500.0, 12.0), word("world", 146.0, 40.0, 499.0, 13.0)],
            ink: [0.1, 0.2, 0.5],
            paper: [1.0, 0.95, 0.9],
            em: None,
        };
        let s = String::from_utf8(editable_layer(&[line])).unwrap();
        // The cover, in the paper colour, before the text.
        let cover = s.find("1 0.95 0.9 rg").expect("paper colour");
        let text = s.find("BT").expect("text object");
        assert!(cover < text && s.contains("h f"), "{s}");
        // Visible text (no `3 Tr`), in the ink colour, the line as one string.
        assert!(!s.contains("Tr") && s.contains("0.1 0.2 0.5 rg") && s.contains("(Hello world) Tj"), "{s}");
        // Sized to the line: lowest bottom 499, highest top 512; level, though the bottoms differ.
        assert!(s.contains(&format!("/PCHelv {} Tf", num(13.0 / BOX_EM))), "{s}");
        assert!(s.contains(&format!("1 0 0 1 100 {} Tm", num(499.0 + DESCENT * 13.0 / BOX_EM))), "{s}");
        let natural = helvetica_width("Hello world", 13.0 / BOX_EM);
        assert!(s.contains(&format!("{} Tz", num((100.0 * 86.0 / natural).clamp(50.0, 200.0)))), "{s}");
        assert!(s.starts_with("q\n") && s.ends_with("Q\n"));
    }

    #[test]
    fn editable_layer_follows_a_turned_page() {
        // A page turned 90°: the line runs up the page.
        let line = PlacedLine {
            words: vec![PlacedWord { text: "Up".into(), origin: [50.0, 10.0], across: [0.0, 30.0], up: [-10.0, 0.0] }],
            ink: [0.0; 3],
            paper: [1.0; 3],
            em: None,
        };
        let s = String::from_utf8(editable_layer(&[line])).unwrap();
        assert!(s.contains(&format!("0 1 -1 0 {} 10 Tm", num(50.0 - DESCENT * 10.0 / BOX_EM))), "{s}");
    }

    /// White, with blocks of colour at `[left, top, right, bottom]`.
    fn painted(w: u32, h: u32, rects: &[([u32; 4], [u8; 3])]) -> Vec<u8> {
        let mut px = vec![255u8; (w * h * 4) as usize];
        for &([l, t, r, b], colour) in rects {
            for y in t..b {
                for x in l..r {
                    let o = ((y * w + x) * 4) as usize;
                    px[o..o + 3].copy_from_slice(&colour);
                }
            }
        }
        px
    }

    fn ocr_word(text: &str, rect: [u32; 4]) -> Word {
        Word { text: text.into(), rect: rect.map(|v| v as f32), confidence: 1.0 }
    }

    fn texts(lines: &[EditableLine]) -> Vec<String> {
        lines.iter().map(|l| l.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")).collect()
    }

    const INK: [u8; 3] = [20, 20, 20];

    /// A printed word: letters filling its box, a little inside the padding OCR adds.
    fn printed(rect: [u32; 4]) -> ([u32; 4], [u8; 3]) {
        ([rect[0] + 2, rect[1] + 2, rect[2] - 2, rect[3] - 2], INK)
    }

    /// The signature row of a delivery note: a signature through "BY", a faded maroon stamp whose
    /// date runs into the printed "TIME:" next to it. Only the clean printed words become text.
    #[test]
    fn handwriting_and_stamps_stay_in_the_image() {
        let words = [
            ("RECEIVED", [10, 40, 90, 60]),
            ("BY", [100, 10, 180, 90]), // the signature's loops stretch the box
            ("DATE:", [200, 40, 250, 60]),
            ("RECEIVED", [300, 40, 380, 60]), // the stamp
            ("2026TIME:", [387, 40, 440, 60]),
            ("Thanks", [520, 40, 580, 60]),
        ];
        let mut ink: Vec<_> = words.iter().map(|(_, r)| printed(*r)).collect();
        ink[3].1 = [110, 60, 80];
        let px = painted(600, 100, &ink);
        let line = Line { words: words.iter().map(|(t, r)| ocr_word(t, *r)).collect() };
        let out = editable_lines(&px, 600, 100, &[line]);
        assert_eq!(texts(&out), ["RECEIVED", "DATE:", "Thanks"]);
        assert!(out.iter().all(|l| l.ink.iter().all(|c| (c - 20.0 / 255.0).abs() < 0.02)), "{out:?}");
    }

    /// A table's header row read as one line: split at the cells' borders, which run past the
    /// letters, but not at a letter's stem that OCR's word box clipped off.
    #[test]
    fn table_cells_split_at_rules_not_at_letter_stems() {
        let words = [
            ("P.O.", [10, 40, 50, 60]),
            ("No.", [58, 40, 90, 60]),
            ("Ship", [130, 40, 170, 60]),
            ("Date", [178, 40, 215, 60]),
            ("Via", [280, 40, 310, 60]),
        ];
        let mut ink: Vec<_> = words.iter().map(|(_, r)| printed(*r)).collect();
        ink.push(([54, 42, 57, 58], INK)); // the N's stem, outside its word's box
        ink.push(([115, 20, 117, 80], INK)); // the border between the cells
        let px = painted(330, 100, &ink);
        let line = Line { words: words.iter().map(|(t, r)| ocr_word(t, *r)).collect() };
        // "Via" is more than 2.5 line heights from "Date": a separate field, though no rule.
        assert_eq!(texts(&editable_lines(&px, 330, 100, &[line])), ["P.O. No.", "Ship Date", "Via"]);
    }

    /// A stamp's date printed over a label, read as a word on a line of its own, is left with the
    /// stamp; a digit OCR didn't read between two words keeps showing (the line splits around it).
    #[test]
    fn stamp_neighbours_on_other_lines_and_unread_ink_stay_visible() {
        let stamp = ([300, 40, 380, 60], [110, 60, 80]);
        let after = [10, 140, 60, 160];
        let days = [100, 140, 150, 160];
        let px = painted(
            400,
            200,
            &[stamp, printed([386, 41, 398, 59]), printed(after), ([70, 143, 80, 157], INK), printed(days), printed([10, 40, 60, 60])],
        );
        let lines = [
            Line { words: vec![ocr_word("Date:", [10, 40, 60, 60]), ocr_word("RECEIVED", stamp.0)] },
            Line { words: vec![ocr_word("THMD", [386, 41, 398, 59])] },
            Line { words: vec![ocr_word("after", after), ocr_word("days", days)] },
        ];
        assert_eq!(texts(&editable_lines(&px, 400, 200, &lines)), ["Date:", "after", "days"]);
    }

    /// A speck read as a letter is left alone, and an em the boxes can't explain (here a dash's
    /// thin ink measured as a whole line's letters) gives way to the boxes' estimate.
    #[test]
    fn specks_and_implausible_ems_are_left_out() {
        let px = painted(400, 100, &[printed([10, 40, 90, 60]), printed([100, 40, 160, 60]), ([200, 49, 280, 51], INK), ([300, 50, 304, 54], INK)]);
        let lines = [
            Line { words: vec![ocr_word("Total", [10, 40, 90, 60]), ocr_word("due", [100, 40, 160, 60])] },
            Line { words: vec![ocr_word("FEES", [198, 40, 282, 60])] },
            Line { words: vec![ocr_word("e", [299, 49, 305, 55])] },
        ];
        let out = editable_lines(&px, 400, 100, &lines);
        assert_eq!(texts(&out), ["Total due", "FEES"]);
        assert!(out[0].em.is_some() && out[1].em.is_none(), "{out:?}");
    }

    /// Never-panic: NaN and huge boxes, short or empty buffers, no words.
    #[test]
    fn editable_lines_survive_bad_input() {
        let px = painted(50, 50, &[printed([5, 10, 40, 30])]);
        let line = Line {
            words: vec![
                ocr_word("ok", [5, 10, 40, 30]),
                Word { text: "nan".into(), rect: [f32::NAN, 0.0, 10.0, 10.0], confidence: 1.0 },
                Word { text: "inf".into(), rect: [0.0, 0.0, f32::INFINITY, 10.0], confidence: 1.0 },
                Word { text: "flat".into(), rect: [10.0, 10.0, 5.0, 5.0], confidence: 1.0 },
            ],
        };
        assert_eq!(texts(&editable_lines(&px, 50, 50, std::slice::from_ref(&line))), ["ok"]);
        // A page-sized box over the word: both are left in the image.
        let mut line = line;
        line.words.push(Word { text: "huge".into(), rect: [0.0, 0.0, 1e9, 1e9], confidence: 1.0 });
        assert!(editable_lines(&px, 50, 50, std::slice::from_ref(&line)).is_empty());
        let _ = editable_lines(&px[..10], 50, 50, std::slice::from_ref(&line));
        let _ = editable_lines(&[], 0, 0, std::slice::from_ref(&line));
        let _ = editable_lines(&px, u32::MAX, u32::MAX, std::slice::from_ref(&line));
        assert!(editable_lines(&px, 50, 50, &[Line::default()]).is_empty());
    }

    #[test]
    fn editable_layer_skips_degenerate_lines() {
        let bad = |words| PlacedLine { words, ink: [0.0; 3], paper: [1.0; 3], em: None };
        let lines = [
            bad(vec![]),
            bad(vec![word("x", 10.0, 0.0, 10.0, 10.0)]),
            bad(vec![word("x", 10.0, 10.0, 10.0, 0.0)]),
            bad(vec![word("x", f64::NAN, 10.0, 10.0, 10.0)]),
            bad(vec![word(" ", 10.0, 10.0, 10.0, 10.0)]),
        ];
        assert_eq!(editable_layer(&lines), b"q\nQ\n");
    }

    /// #103: every release package's layout puts the models where the installed app looks.
    /// Mirrors packaging/{windows/package.ps1,macos/package.sh,linux/package.sh,freebsd/package.sh}.
    #[test]
    fn finds_the_models_in_every_release_package_layout() {
        let base = std::env::temp_dir().join(format!("pdfcraft-ocr-layouts-{}", std::process::id()));
        let layouts = [
            ("windows", "PdfCraft", "PdfCraft/models"),
            ("macos", "PdfCraft.app/Contents/MacOS", "PdfCraft.app/Contents/Resources/models"),
            ("linux-deb-rpm", "usr/bin", "usr/share/pdfcraft/models"),
            ("linux-tar", "pdfcraft-1.0.0-linux-x86_64/bin", "pdfcraft-1.0.0-linux-x86_64/share/pdfcraft/models"),
            ("appimage", "PdfCraft.AppDir/usr/bin", "PdfCraft.AppDir/usr/share/pdfcraft/models"),
            ("flatpak", "app/bin", "app/share/pdfcraft/models"),
            ("freebsd", "usr/local/bin", "usr/local/share/pdfcraft/models"),
        ];
        for (name, exe_dir, models_dir) in layouts {
            let root = base.join(name);
            let (exe_dir, models_dir) = (root.join(exe_dir), root.join(models_dir));
            std::fs::create_dir_all(&exe_dir).unwrap();
            std::fs::create_dir_all(&models_dir).unwrap();
            let found = || Models::dirs_beside_exe(&exe_dir).iter().find_map(|d| Models::in_dir(d));
            assert_eq!(found(), None, "{name}: no models yet");
            for file in [DETECTION_MODEL, RECOGNITION_MODEL] {
                std::fs::write(models_dir.join(file), b"model").unwrap();
            }
            let models = found().unwrap_or_else(|| panic!("{name}: models in {} not found", models_dir.display()));
            assert_eq!(models.detection.canonicalize().unwrap(), models_dir.join(DETECTION_MODEL).canonicalize().unwrap(), "{name}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A word the recogniser isn't sure of stays in the image: the line splits around it.
    #[test]
    fn unsure_words_stay_in_the_image() {
        let words = [("Area", [10, 40, 50, 60]), ("L?de", [58, 40, 95, 60]), ("Phone", [103, 40, 150, 60])];
        let ink: Vec<_> = words.iter().map(|(_, r)| printed(*r)).collect();
        let px = painted(200, 100, &ink);
        let mut line = Line { words: words.iter().map(|(t, r)| ocr_word(t, *r)).collect() };
        line.words[1].confidence = 0.6;
        assert_eq!(texts(&editable_lines(&px, 200, 100, &[line])), ["Area", "Phone"]);
    }

    /// Confidences read again go to the right words: in order when the readings split alike,
    /// by text when they don't, the line's lowest for a word not found.
    #[test]
    fn confidences_follow_their_words() {
        let words = |t: &[&str]| t.iter().map(|t| ocr_word(t, [0, 0, 10, 10])).collect::<Vec<_>>();
        let conf = |w: &[Word]| w.iter().map(|w| w.confidence).collect::<Vec<_>>();
        let mut w = words(&["MIX", "&MATCH"]);
        assign_confidences(&mut w, &[("MIX".into(), 0.99), ("&MATCH".into(), 0.64)]);
        assert_eq!(conf(&w), [0.99, 0.64]);
        let mut w = words(&["MIX", "&", "MATCH", "Atfr"]);
        assign_confidences(&mut w, &[("MIX".into(), 0.99), ("&MATCH".into(), 0.64), ("Atfr".into(), 0.5)]);
        assert_eq!(conf(&w), [0.99, 0.5, 0.5, 0.5]);
        let mut w = words(&["A"]);
        assign_confidences(&mut w, &[]);
        assert_eq!(conf(&w), [1.0], "nothing read again: left as it was measured");
    }

    /// The message tells end users what to do; release builds have no cargo.
    #[test]
    fn the_alphabet_is_the_trained_one() {
        let chars: Vec<char> = ALPHABET.chars().collect();
        assert_eq!(chars.len(), 96);
        // The euro sign follows the ASCII punctuation (class 44; 0 is the CTC blank), not a second E.
        assert_eq!(chars.iter().position(|&c| c == '€'), Some(43));
        assert_eq!(chars.iter().filter(|&&c| c == 'E').count(), 1);
    }

    #[test]
    fn missing_models_message_is_for_end_users() {
        let message = OcrError::NoModels.to_string();
        assert!(!message.contains("cargo") && message.contains("PDFCRAFT_MODELS"), "{message}");
    }

    /// End to end on rendered text, when the models are installed (`cargo xtask models`).
    #[test]
    fn reads_rendered_text() {
        let Some(models) = Models::find() else {
            eprintln!("skipped: OCR models not installed");
            return;
        };
        let ocr = Ocr::load(&models).unwrap();
        let (w, h, px) = test_image::hello();
        let lines = ocr.recognize(&px, w, h).unwrap();
        let text: Vec<String> = lines.iter().map(Line::text).collect();
        assert!(text.iter().any(|t| t.contains("HELL") && t.contains("WOR")), "{text:?}");
        let word = lines.iter().flat_map(|l| &l.words).find(|w| w.text.contains("HELL")).unwrap();
        assert!(word.rect[0] >= 10.0 && word.rect[0] < 60.0, "{word:?}");
        assert!(word.confidence > 0.0 && word.confidence <= 1.0, "{word:?}");
    }

    mod test_image {
        /// "HELLO WORLD" drawn with blocky 5×7 letters, 4 pixels per dot, black on white.
        pub fn hello() -> (u32, u32, Vec<u8>) {
            const GLYPHS: &[(char, [&str; 7])] = &[
                ('H', ["X...X", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"]),
                ('E', ["XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "XXXXX"]),
                ('L', ["X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"]),
                ('O', [".XXX.", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."]),
                ('W', ["X...X", "X...X", "X...X", "X.X.X", "X.X.X", "XX.XX", "X...X"]),
                ('R', ["XXXX.", "X...X", "X...X", "XXXX.", "X.X..", "X..X.", "X...X"]),
                ('D', ["XXXX.", "X...X", "X...X", "X...X", "X...X", "X...X", "XXXX."]),
            ];
            let text = "HELLO WORLD";
            let (dot, x0, y0) = (4u32, 40u32, 40u32);
            let (w, h) = (x0 * 2 + text.len() as u32 * 6 * dot, y0 * 2 + 7 * dot);
            let mut px = vec![255u8; (w * h * 4) as usize];
            for (i, c) in text.chars().enumerate() {
                let Some((_, rows)) = GLYPHS.iter().find(|g| g.0 == c) else { continue };
                for (ry, row) in rows.iter().enumerate() {
                    for (rx, b) in row.bytes().enumerate() {
                        if b != b'X' {
                            continue;
                        }
                        for dy in 0..dot {
                            for dx in 0..dot {
                                let x = x0 + (i as u32 * 6 + rx as u32) * dot + dx;
                                let y = y0 + ry as u32 * dot + dy;
                                let o = ((y * w + x) * 4) as usize;
                                px[o..o + 3].copy_from_slice(&[0, 0, 0]);
                            }
                        }
                    }
                }
            }
            (w, h, px)
        }
    }
}
