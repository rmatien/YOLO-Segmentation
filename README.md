### YOLO Segmentation

- Dataset menggunakan 29 Image yang dicapture dari rekaman CCTV
- Notasi polygon menggunakan LabelMe
- Hasil JSON LabelMe di convert menjadi YOLO YAML menggunakan labelme2yolov8
- Train Dataset menggunakan perintah berikut

  ```yolo task=segment mode=train imgsz=640 data="PlatNomorDataset\dataset.yaml" epochs=100 name=platnomor-seg```

-  Untuk melihat hasilnya jalankan script **platnomor-seg.py**
