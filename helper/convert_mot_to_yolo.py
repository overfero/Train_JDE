import os
import numpy as np
from PIL import Image

def convert_mot_to_yolo(seq_path, out_dir):
    """
    Mengonversi anotasi MOT17/MOT20 ke format YOLO biasa (1 class: person).
    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    YOLO format: <class> <x_center> <y_center> <width> <height> (semua dinormalisasi 0-1)
    """
    img_dir = os.path.join(seq_path, 'img1')
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
    
    if not os.path.exists(gt_path):
        return

    # Buat folder output untuk images dan labels
    seq_name = os.path.basename(seq_path)
    # Kami akan meniru struktur direktori YOLO standar (yaitu images/train, labels/train)
    # MOT hanya memiliki ground truth untuk set 'train', jadi kita masukkan ke folder 'train'
    out_img_dir = os.path.join(out_dir, 'images', 'train')
    out_lbl_dir = os.path.join(out_dir, 'labels', 'train')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Ambil resolusi gambar dari seqinfo.ini (lebih cepat daripada baca PIL per folder)
    img_width, img_height = 0, 0
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path, 'r') as f:
            for line in f:
                if line.startswith('imWidth='):
                    img_width = int(line.split('=')[1].strip())
                elif line.startswith('imHeight='):
                    img_height = int(line.split('=')[1].strip())
    
    if img_width == 0 or img_height == 0:
        # Fallback kalau seqinfo gak ada
        img_files = sorted(os.listdir(img_dir))
        if not img_files:
            return
        sample_img_path = os.path.join(img_dir, img_files[0])
        with Image.open(sample_img_path) as img:
            img_width, img_height = img.size

    # Baca anotasi gt.txt
    try:
        data = np.loadtxt(gt_path, delimiter=',')
    except Exception as e:
        print(f"Error membaca {gt_path}: {e}")
        return
        
    if len(data) == 0:
        return
    
    # Kelompokkan data berdasarkan frame
    frames = np.unique(data[:, 0])
    
    # Pre-calculate offset untuk menghindari nama file berbenturan antar sequence yang berbeda
    # Kita prepend nama sequence ke nama file
    
    for frame in frames:
        frame_data = data[data[:, 0] == frame]
        
        # MOT frame biasanya 6 digit
        frame_str = f"{int(frame):06d}"
        
        # Penamaan file output: {seq_name}_{frame_str}
        out_filename_base = f"{seq_name}_{frame_str}"
        img_out_filename = f"{out_filename_base}.jpg"
        lbl_out_filename = f"{out_filename_base}.txt"
        
        # Copy/Symlink gambar asli
        # Beberapa seq file ekstensinya bisa berbeda, jadi kita cek file yang ada
        orig_img_filename = f"{frame_str}.jpg" 
        src_img = os.path.join(img_dir, orig_img_filename)
        
        dst_img = os.path.join(out_img_dir, img_out_filename)
        
        # Jangan buat ulang symlink kalau udah ada
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
            
        # Kumpulkan bounding box frame ini
        yolo_lines = []
        for row in frame_data:
            # MOT Format (https://arxiv.org/pdf/2003.09003.pdf) / MOT17 Format
            # MOT20 sedikit lebih ribet namun strukturnya mirip:
            # Frame, ID, bbox left, bbox top, bbox width, bbox height, conf, class, visibility
            
            # Cek panjang kolom untuk antisipasi error parsing
            if len(row) < 7:
                continue
                
            conf = row[6]
            # Untuk ground truth, conf bisa berarti ignore flag (0 = ignore, 1 = active)
            # Kolom 8 (idx 7) adalah class ID (1 = pedestrian di MOT16/MOT17/MOT20)
            # Kolom 9 (idx 8) adalah visibility
            
            # Filter hanya active, class pejalan kaki, dsb
            # Jika dataset MOT lama yang formatnya beda, kita sesuaikan
            if len(row) >= 8:
                class_id_mot = row[7]
                vis = row[8] if len(row) >= 9 else 1.0
                
                # Sesuai standard JDE / MOT: 
                # 1: pedestrian, 2: person on vehicle, 7: static person dll. 
                # Biasa training pedestrian tracking = class 1 (dan mungkin 2,7,8).
                # Di sini hanya class 1.
                if conf == 0 or class_id_mot != 1:
                    continue
                if vis < 0.1: # Skip heavily occluded/invisible
                    continue

            bb_left = row[2]
            bb_top = row[3]
            bb_width = row[4]
            bb_height = row[5]

            # Hitung Center
            x_center = bb_left + (bb_width / 2.0)
            y_center = bb_top + (bb_height / 2.0)

            # Normalisasi
            x_center /= img_width
            y_center /= img_height
            w = bb_width / img_width
            h = bb_height / img_height

            # Clip nilai antara 0-1
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            # YOLO Format: class x_center y_center width height
            # Hanya 1 class (Person) -> class_id = 0
            if w > 0 and h > 0:
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
        # Tulis label txt
        if yolo_lines:
            lbl_path = os.path.join(out_lbl_dir, lbl_out_filename)
            with open(lbl_path, 'w') as f:
                f.writelines(yolo_lines)


def main():
    base_dir = '/kaggle/working/Train_JDE'
    out_dir = os.path.join(base_dir, 'YOLO_Dataset')
    
    datasets = ['MOT17', 'MOT20']
    
    for ds in datasets:
        ds_path = os.path.join(base_dir, ds)
        if not os.path.exists(ds_path):
            print(f"Folder {ds_path} tidak ditemukan, skip.")
            continue
            
        print(f"Memproses {ds}...")
        
        # Umumnya dataset training (yang ada ground truth) ada di folder 'train'
        train_path = os.path.join(ds_path, 'train')
        if not os.path.exists(train_path):
            print(f"  Tidak ada folder train di {ds}, mencoba membaca semua subfolder.")
            # Fallback (Misal folder diekstrak tanpa /train/)
            train_path = ds_path
            
        for seq in os.listdir(train_path):
            seq_path = os.path.join(train_path, seq)
            if os.path.isdir(seq_path) and os.path.exists(os.path.join(seq_path, 'img1')):
                print(f"  -> Konversi {seq}...")
                convert_mot_to_yolo(seq_path, out_dir)
                    
    print(f"\nSelesai! Dataset YOLO tersedia di: {out_dir}")

if __name__ == '__main__':
    main()
