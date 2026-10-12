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
    #[error("the text recognition models are not installed (reinstall PdfCraft, or set PDFCRAFT_MODELS to the folder that holds them)")]
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

/// A recognised word: its text and its box in image pixels `[left, top, right, bottom]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub rect: [f32; 4],
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
        Ok(Ocr { engine })
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
            .flatten()
            .map(|line| Line {
                words: line
                    .words()
                    .map(|w| {
                        let r = w.bounding_rect();
                        Word { text: w.to_string(), rect: [r.left() as f32, r.top() as f32, r.right() as f32, r.bottom() as f32] }
                    })
                    .filter(|w| !w.text.trim().is_empty())
                    .collect(),
            })
            .filter(|l| !l.words.is_empty())
            .collect())
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
    Some(Word { text: text.into(), rect: [tight[0], (baseline - em) as f32, tight[2], baseline as f32] })
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
        let w = PlacedWord::place(&Word { text: "Hello".into(), rect: [100.0, 200.0, 300.0, 240.0] }, to_user);
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
        let w = PlacedWord::place(&Word { text: "Up".into(), rect: [0.0, 0.0, 50.0, 10.0] }, to_user);
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
