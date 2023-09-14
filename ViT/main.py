import torch
from pathlib import Path
import requests
import torchvision
from super_repo import data_setup, engine, utils
from super_repo.utils import plot_loss_curves
from config import *
from model import *
from utils import *

def runner(pretrained = False):

    if not pretrained:

        train_dataloader, test_dataloader, class_names = create_dataloaders(TRAIN_DIR, TEST_DIR, MANUAL_TRANSFORMS, BATCH_SIZE,NUM_WORKERS)

        # Create a random tensor with same shape as a single image
        random_image_tensor = torch.randn(1, 3, 224, 224) # (batch_size, color_channels, height, width)

        # Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
        vit = ViT(num_classes=len(class_names))

        # Pass the random image tensor to our ViT instance
        vit(random_image_tensor)

        # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper 
        optimizer = torch.optim.Adam(params=vit.parameters(), 
                                    lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                                    betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                    weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

        # Setup the loss function for multi-class classification
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the model and save the training results to a dictionary
        results = engine.train(model=vit,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=2,
                            device=DEVICE)
        
        plot_loss_curves(results)
    
    elif pretrained:

        # 1. Get pretrained weights for ViT-Base
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # requires torchvision >= 0.13, "DEFAULT" means best available

        # 2. Setup a ViT model instance with pretrained weights
        pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(DEVICE)

        # 3. Freeze the base parameters
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False
            

        pretrained_vit.heads = nn.Linear(in_features=768, out_features=3).to(DEVICE)
        # pretrained_vit # uncomment for model output 

        pretrained_vit_transforms = pretrained_vit_weights.transforms()

        train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=TRAIN_DIR,
                                                                                                        test_dir=TEST_DIR,
                                                                                                        transform=pretrained_vit_transforms,
                                                                                                        batch_size=32) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)


        # Create optimizer and loss function
        optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                                    lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the classifier head of the pretrained ViT feature extractor model
        pretrained_vit_results = engine.train(model=pretrained_vit,
                                            train_dataloader=train_dataloader_pretrained,
                                            test_dataloader=test_dataloader_pretrained,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            epochs=3,
                                            device=DEVICE)
        
        plot_loss_curves(pretrained_vit_results)
        print()

        # Save the model
        utils.save_model(model=pretrained_vit,
                    target_dir="models",
                    model_name="08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth")
        
        # Get the model size in bytes then convert to megabytes
        pretrained_vit_model_size = Path("models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth").stat().st_size // (1024*1024) # division converts bytes to megabytes (roughly) 
        print(f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB")