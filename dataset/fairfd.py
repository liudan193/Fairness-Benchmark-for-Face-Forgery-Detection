from copy import deepcopy
import os
import torch
import cv2
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from PIL import Image

class FairFD(Dataset):
    """
    FairFD dataset Class
    """

    def __init__(self, config, root_path):
        # pre-check
        super(FairFD, self).__init__()
        print(f"Loading data from FairFD ...")
        self.config = config
        self.root = root_path
        self.image_list, self.label_list = self.__get_images_rfw()
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }
        self.transform = self.init_data_aug_method()
        print(f"Data from '{root_path}' loaded.")
        print(f"Dataset contains {len(self.image_list)} images.\n")

    def __get_images_rfw(self):
        if (("FaceSwap" in self.root) or ("SimSwap" in self.root) or ("FaceReen" in self.root) or
                ("Dual_Generator_Face_Reen" in self.root) or ("MaskGan" in self.root) or
                ("StyGAN" in self.root) or ("SDSwap" in self.root) or ("StarSwap" in self.root)):
            print("=" * 50)
            fake_images = []
            for id_path in os.listdir(self.root):
                temp_image_names = os.listdir(join(self.root, id_path))
                fake_images.extend([join(id_path, image_name) for image_name in temp_image_names])
            fake_tgts = [torch.tensor(1)] * len(fake_images)
            print(f"fake: {len(fake_tgts)}")
            return fake_images, fake_tgts
        else:
            real_images = []
            for id_path in os.listdir(self.root):
                temp_image_names = os.listdir(join(self.root, id_path))
                real_images.extend([join(id_path, image_name) for image_name in temp_image_names])
            real_tgts = [torch.tensor(0)] * len(real_images)
            print(f"real: {len(real_tgts)}")
            return real_images, real_tgts

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'],
                                           contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'],
                               quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ],
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def __len__(self):
        return len(self.label_list)

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def __getitem__(self, index):
        image_path = join(self.root, self.image_list[index])

        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation

        # Load mask and landmark (if needed)
        mask = None
        landmarks = None

        # Do transforms
        if self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask)
        else:
            image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)

        label = self.label_list[index]
        return image_trans, label, landmarks_trans, mask_trans

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Create a dictionary of arguments
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if landmarks[0] is not None:
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks[0] is not None:
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

