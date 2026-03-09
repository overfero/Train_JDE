import kagglehub

# Ganti 'USERNAME' dengan username Kaggle-mu (bisa dilihat di URL profilmu atau pojok kiri bawah)
# Ganti 'DATASET_SLUG' dengan nama kecil bersambung, misalnya 'crowdhuman-extracted-yolo'
handle = 'fehruputra/mot-dataset' 

# Path langsung menuju folder dataset kamu
local_dataset_dir = '/kaggle/working/Train_JDE/mot'

# Upload folder tersebut menjadi dataset Kaggle (versi baru)
kagglehub.dataset_upload(handle, local_dataset_dir, version_notes='Uploaded extracted dataset')

print("Proses upload dataset berhasil di-request!")
