"""Veri modülü: manifest -> öznitelik -> DataLoader.

İş akışı:
1. **Manifest** (JSONL) her satırda anonimleştirilmiş bir ``Sample``
   içerir (yol + etiket). Bkz. ``data/schema.py``.
2. ``FeatureExtractor`` her örneği modalite özniteliklerine çevirir
   (yüz işaretçisi / MFCC+eGeMAPS / BERTurk token). Ağır bağımlılıklar
   tembel yüklenir; enjekte edilebilir olduğundan test/geliştirmede
   sahte bir çıkarıcı kullanılabilir.
3. ``collate_features`` değişken uzunlukları doldurur (pad) ve bir
   ``ModalityBatch`` üretir.
4. ``build_dataloaders`` sınıf-dengeli (stratified) eğitim/doğrulama
   ayrımı yapar ve ``DataLoader`` döndürür.

Gerçek klinik veri henüz yokken hattı uçtan uca doğrulamak için
``SyntheticDataset`` kullanılabilir (bkz. ``training/train.py`` smoke).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..modalities.wearable.features import NUM_WEARABLE_CHANNELS
from .batch import ModalityBatch, SampleFeatures
from .modalities_config import DataConfig
from .schema import DiagnosisLabel, Sample

# İkili deney için varsayılan sınıf sırası (kontrol = 0)
DEFAULT_CLASSES: tuple[str, ...] = (
    DiagnosisLabel.KONTROL.value,
    DiagnosisLabel.DEPRESYON.value,
)

# Çok-hastalık deney tasarımı için kanonik sınıf sırası (kontrol = 0).
# Manifest'te bulunan etiketlere göre alt kümesi kullanılır.
DIAGNOSIS_CLASSES: tuple[str, ...] = (
    DiagnosisLabel.KONTROL.value,
    DiagnosisLabel.DEPRESYON.value,
    DiagnosisLabel.ANKSIYETE.value,
    DiagnosisLabel.BIPOLAR.value,
    DiagnosisLabel.SIZOFRENI.value,
)


def infer_classes(samples: Sequence[Sample]) -> list[str]:
    """Manifest örneklerinde bulunan etiketleri kanonik sırada döndürür.

    Kontrol her zaman ilk sırada (indeks 0) tutulur; böylece ikili/çok-
    sınıflı metrik semantiği (pozitif = tanı) korunur.
    """
    present = {s.label.value for s in samples if s.label is not None}
    ordered = [c for c in DIAGNOSIS_CLASSES if c in present]
    # Kanonik listede olmayan etiketleri de sona ekle (ör. 'belirsiz').
    extra = sorted(present - set(DIAGNOSIS_CLASSES))
    return ordered + extra


class LabelEncoder:
    """Etiket dizisi <-> tam sayı indeksi eşlemesi."""

    def __init__(self, classes: Sequence[str] = DEFAULT_CLASSES) -> None:
        self.classes = list(classes)
        self._to_idx = {c: i for i, c in enumerate(self.classes)}

    def encode(self, label: str | DiagnosisLabel) -> int:
        key = label.value if isinstance(label, DiagnosisLabel) else label
        if key not in self._to_idx:
            raise KeyError(
                f"Bilinmeyen etiket '{key}'. Tanımlı sınıflar: {self.classes}"
            )
        return self._to_idx[key]

    def decode(self, index: int) -> str:
        return self.classes[index]

    def __len__(self) -> int:
        return len(self.classes)


# --------------------------------------------------------------------------- #
# Öznitelik çıkarımı
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Bir ``Sample``'ı modalite öznitelik tensörlerine çevirir.

    Ağır bağımlılıklar (mediapipe, librosa, transformers) yalnızca ilgili
    modalite etkinse ve ilk kullanımda yüklenir. Bu sınıf test için
    alt sınıflanabilir/taklit edilebilir (dependency injection).
    """

    def __init__(
        self,
        cfg: DataConfig | None = None,
        modalities: Sequence[str] = ("vision", "audio", "text", "wearable"),
        data_root: Path | None = None,
    ) -> None:
        self.cfg = cfg or DataConfig()
        self.modalities = tuple(modalities)
        # Manifest'teki göreli yollar bu köke göre çözülür.
        from ..config import get_settings

        self.data_root = Path(data_root) if data_root else get_settings().data_dir
        self._tokenizer = None  # tembel

    def _resolve(self, path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.data_root / path

    # -- metin tokenizer (BERTurk) -- #
    @property
    def tokenizer(self):  # noqa: ANN201
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Metin modalitesi için 'transformers' gerekli. "
                    "Kurulum: pip install 'psikiyatri-ai[text]'"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_model_name)
        return self._tokenizer

    def extract(self, sample: Sample) -> dict[str, torch.Tensor]:
        """Örnekten etkin modalitelerin özniteliklerini çıkarır.

        ``features_path`` verilmişse ham medya yerine önceden çıkarılmış
        öznitelikler (.npz) doğrudan yüklenir.
        """
        if sample.features_path is not None:
            return self._load_precomputed(self._resolve(sample.features_path))

        feats: dict[str, torch.Tensor] = {}
        if "vision" in self.modalities and sample.video_path is not None:
            feats["vision"] = self._extract_vision(self._resolve(sample.video_path))
        if "audio" in self.modalities and sample.audio_path is not None:
            mfcc, egemaps = self._extract_audio(self._resolve(sample.audio_path))
            feats["audio_mfcc"] = mfcc
            if egemaps is not None:
                feats["audio_egemaps"] = egemaps
        if "wearable" in self.modalities and sample.wearable_path is not None:
            feats["wearable"] = self._extract_wearable(self._resolve(sample.wearable_path))
        if "text" in self.modalities and sample.transcript:
            ids, mask = self._extract_text(sample.transcript)
            feats["text_input_ids"] = ids
            feats["text_attention_mask"] = mask
        return feats

    def _load_precomputed(self, path: Path) -> dict[str, torch.Tensor]:
        """Önceden çıkarılmış öznitelikleri (.npz) tensör sözlüğüne yükler."""
        import numpy as np

        if not path.exists():
            raise FileNotFoundError(f"Öznitelik dosyası bulunamadı: {path}")
        data = np.load(path)
        int_keys = {"text_input_ids", "text_attention_mask"}
        feats: dict[str, torch.Tensor] = {}
        for key in data.files:
            tensor = torch.from_numpy(data[key])
            feats[key] = tensor.long() if key in int_keys else tensor.float()
        return feats

    def _extract_vision(self, video_path: Path) -> torch.Tensor:
        from ..modalities.vision.preprocess import extract_face_landmarks, sample_frames

        frames = sample_frames(video_path, fps=self.cfg.frame_sample_fps)
        landmarks = [extract_face_landmarks(f).landmarks for f in frames]
        if not landmarks:
            from ..modalities.vision.preprocess import NUM_FACE_LANDMARKS

            landmarks = [torch.zeros(NUM_FACE_LANDMARKS, 3).numpy()]
        return torch.from_numpy(_stack_np(landmarks)).float()  # (T, 468, 3)

    def _extract_audio(self, audio_path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
        from ..modalities.audio.features import extract_acoustic_features

        af = extract_acoustic_features(audio_path, with_egemaps=self.cfg.use_egemaps)
        mfcc = torch.from_numpy(af.mfcc.T).float()  # (T, 40)
        egemaps = torch.from_numpy(af.egemaps).float() if af.egemaps is not None else None
        return mfcc, egemaps

    def _extract_wearable(self, wearable_path: Path) -> torch.Tensor:
        from ..modalities.wearable.features import extract_wearable_features

        wf = extract_wearable_features(
            wearable_path, sample_rate=self.cfg.wearable_sample_rate
        )
        return torch.from_numpy(wf.series).float()  # (T, C)

    def _extract_text(self, transcript: str) -> tuple[torch.Tensor, torch.Tensor]:
        from ..modalities.text.preprocess import normalize_text

        enc = self.tokenizer(
            normalize_text(transcript),
            truncation=True,
            max_length=self.cfg.text_max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def _stack_np(arrays):  # noqa: ANN001, ANN202
    import numpy as np

    return np.stack(arrays, axis=0)


# --------------------------------------------------------------------------- #
# Dataset'ler
# --------------------------------------------------------------------------- #
class ManifestDataset(Dataset):
    """JSONL manifest'inden örnek okuyan dataset."""

    def __init__(
        self,
        samples: Sequence[Sample],
        extractor: FeatureExtractor,
        label_encoder: LabelEncoder,
    ) -> None:
        self.samples = list(samples)
        self.extractor = extractor
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleFeatures:
        sample = self.samples[idx]
        if sample.label is None:
            raise ValueError(f"Etiketsiz örnek eğitimde kullanılamaz: {sample.session_id}")
        feats = self.extractor.extract(sample)
        return SampleFeatures(label=self.label_encoder.encode(sample.label), **feats)


@dataclass
class SyntheticDataset(Dataset):
    """Rastgele öznitelik üreten dataset — hat doğrulama/smoke için.

    Gerçek veri hazır olana kadar eğitim döngüsünün uçtan uca çalıştığını
    kanıtlamak için kullanılır (dosya/ağır bağımlılık gerektirmez).
    """

    size: int = 32
    num_classes: int = 2
    modalities: tuple[str, ...] = ("vision", "audio")
    t_vision: int = 8
    t_audio: int = 50
    t_wearable: int = 60
    text_len: int = 24
    vocab_size: int = 32_000
    seed: int = 42
    _labels: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        g = torch.Generator().manual_seed(self.seed)
        self._labels = torch.randint(
            0, self.num_classes, (self.size,), generator=g
        ).tolist()
        self._gen = torch.Generator().manual_seed(self.seed + 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> SampleFeatures:
        g = self._gen
        feats: dict[str, torch.Tensor] = {}
        if "vision" in self.modalities:
            feats["vision"] = torch.randn(self.t_vision, 468, 3, generator=g)
        if "audio" in self.modalities:
            feats["audio_mfcc"] = torch.randn(self.t_audio, 40, generator=g)
            feats["audio_egemaps"] = torch.randn(88, generator=g)
        if "wearable" in self.modalities:
            feats["wearable"] = torch.randn(
                self.t_wearable, NUM_WEARABLE_CHANNELS, generator=g
            )
        if "text" in self.modalities:
            feats["text_input_ids"] = torch.randint(
                0, self.vocab_size, (self.text_len,), generator=g
            )
            feats["text_attention_mask"] = torch.ones(self.text_len, dtype=torch.long)
        return SampleFeatures(label=self._labels[idx], **feats)


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #
def _collect(samples: Sequence[SampleFeatures], attr: str) -> list[torch.Tensor] | None:
    vals = [getattr(s, attr) for s in samples]
    if all(v is None for v in vals):
        return None
    if any(v is None for v in vals):
        raise ValueError(
            f"Karışık modalite: '{attr}' bazı örneklerde eksik. Eğitimde bir "
            "batch içindeki tüm örnekler aynı modalite kümesine sahip olmalıdır."
        )
    return vals


def collate_features(samples: Sequence[SampleFeatures]) -> tuple[ModalityBatch, torch.Tensor]:
    """``SampleFeatures`` listesini doldurulmuş bir ``ModalityBatch``'e çevirir."""
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)

    vision = _collect(samples, "vision")
    mfcc = _collect(samples, "audio_mfcc")
    egemaps = _collect(samples, "audio_egemaps")
    wearable = _collect(samples, "wearable")
    ids = _collect(samples, "text_input_ids")
    mask = _collect(samples, "text_attention_mask")

    batch = ModalityBatch(
        vision=pad_sequence(vision, batch_first=True) if vision else None,
        audio_mfcc=pad_sequence(mfcc, batch_first=True) if mfcc else None,
        audio_egemaps=torch.stack(egemaps) if egemaps else None,
        wearable=pad_sequence(wearable, batch_first=True) if wearable else None,
        text_input_ids=pad_sequence(ids, batch_first=True) if ids else None,
        text_attention_mask=(
            pad_sequence(mask, batch_first=True) if mask else None
        ),
    )
    return batch, labels


# --------------------------------------------------------------------------- #
# Manifest yükleme + DataLoader kurulumu
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> list[Sample]:
    """JSONL manifest'ini ``Sample`` listesine yükler."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest bulunamadı: {path}")
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(Sample.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Manifest satırı {line_no} geçersiz: {exc}") from exc
    return samples


def build_dataloaders(
    manifest_path: Path,
    *,
    cfg: DataConfig | None = None,
    classes: Sequence[str] = DEFAULT_CLASSES,
    modalities: Sequence[str] = ("vision", "audio", "text", "wearable"),
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    data_root: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Manifest'ten sınıf-dengeli eğitim/doğrulama DataLoader'ları kurar."""
    cfg = cfg or DataConfig()
    samples = load_manifest(manifest_path)
    encoder = LabelEncoder(classes)
    extractor = FeatureExtractor(cfg, modalities=modalities, data_root=data_root)

    labels = [encoder.encode(s.label) for s in samples if s.label is not None]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_split,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    def _loader(subset: Sequence[Sample], shuffle: bool) -> DataLoader:
        ds = ManifestDataset(subset, extractor, encoder)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_features,
            num_workers=num_workers,
        )

    return _loader(train_samples, True), _loader(val_samples, False)


__all__ = [
    "DEFAULT_CLASSES",
    "DIAGNOSIS_CLASSES",
    "FeatureExtractor",
    "LabelEncoder",
    "ManifestDataset",
    "SyntheticDataset",
    "build_dataloaders",
    "collate_features",
    "infer_classes",
    "load_manifest",
]
