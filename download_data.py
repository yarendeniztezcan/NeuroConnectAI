from nilearn.datasets import fetch_abide_pcp

dataset = fetch_abide_pcp(
    data_dir="./data",
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=["rois_cc200"],
    quality_checked=True,
    n_subjects=None
)

print("\nVeri indirme tamamlandı.")
print("Katılımcı sayısı:", len(dataset.rois_cc200))
print("İlk veri boyutu:", dataset.rois_cc200[0].shape)

dataset.phenotypic.to_csv(
    "data/participants.csv",
    index=False
)

print(dataset.phenotypic.head())
