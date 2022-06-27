Set Up Environment:  
1. conda env update --file environment.yml --prune  
2. pip install protobuf==3.19.0 (For tensorboardX)

---

Download Dataset:  
KITTI: ```wget -i <path to KITTI txt file> -P <path for saving KITTI>```

Convert all .png to .jpg:
```find <path to KITTI> -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'```

---

trainer.py line53 interpolation antialias  
UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.

---

```mono_dataset -> __get_item__() -> self.get_color() or self.get_colors() for cityscapes -> pil_loader() -> open file```

---

frame_idxs == "s" means the frame of the other side.

---

## Things to do

- [x] Train
- [x] Real Time Video Prediction

### Small changes to make

- [ ] Print loss at the end of training