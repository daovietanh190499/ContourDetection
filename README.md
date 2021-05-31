### Huấn luyện với tập Pascal VOC

Thay đổi các tham số trong file train.py

```
self.images_path = "/content/drive/MyDrive/convert_dataset/images/"
self.ctns_path = "/content/drive/MyDrive/convert_dataset/ctns/"
self.train_path = "/content/drive/MyDrive/train.txt"
self.val_path = "/content/drive/MyDrive/val.txt"
self.model_save_path = "/content/drive/MyDrive/model_seg/"
self.model_save_name = "cedn_epoch_9.pth"
```

Huấn luyện bằng đối tượng Trainer

```
from CEDN import CEDN
trainer = Trainer(CEDN())
trainer.train()
```
