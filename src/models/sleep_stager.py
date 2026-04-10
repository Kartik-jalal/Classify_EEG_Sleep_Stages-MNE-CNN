"""
Docstring for src.models.__init__.py.sleep_stager_chambon_2018

This module defines the SleepStagerChambon2018 class, which is a PyTorch based CCN 
architecture implementation used for sleep staging tasks. The architecture 
implementation was described in Chambon at al. (2018). However, in this model
instead of using ReLU to introduce non-linearity we use LeakyRelu to prevent 
dying neurons with the use of a leak.

##### Author: Kartik M. Jalal
##### Last Updated: 03-04-2026
"""


import torch
from torch import nn

class SleepStagerChambon2018(nn.Module):
    """
    PyTorch implementation of the sleep staging architecture described in 
    Chambon at al. (2018).

    This Convolutional Neural Network (ConvNet) is designed specifically for
    multivariate EEG time-series data. It uses a two-step feature extraction process:
        1. Spatial Convolutions: Extracts features across different physical 
            electrodes.
        2. Temporal Convolutions: Extracts time-based features (like sleep spindles)
            from the virtual spatial channels.
    
    References
    ----------
    Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., & Gramfort, A. (2018).
    A deep learning architecture for temporal sleep stage classification using 
    multivariate and multimodal time series. IEEE Transactions on Neural Systems 
    and Rehabilitation Engineering, 26(4), 758-769.

    Parameters
    ----------
    - eeg_epoch_duration : float
        The duration of each labeled EEG epoch
    - n_channels : int
        The number of physical EEG channels in the input data (e.g., 2 for FPZ-Cz and
        Pz-Oz).
    - sfreq : float
        The sampling frequency of the EEF data in Hz (e.g.,a sampling rate of 100.0 Hz
        means a sample is recorded every 0.01 seconds or 0.01 x 1000 = 10 milliseconds).
    - n_classes : int, optional
        The number of sleep stages to predict. Default is 5 (W, N1, N2, N3, R).
    - n_spatial_filters : int, optional
        The number of feature maps (filters) to learn in the spatial convolutional
        layers. Higher number increase model capacity but risk overfitting. Deafult 
        is 8.
    - n_temporal_filters_l1 : int, optional
        The number of feature maps (filters) to learn in the temporal convolutional
        layer 1. Higher number increase model capacity but risk overfitting. Deafult 
        is 18.
    - n_temporal_filters_l2 : int, optional
        The number of feature maps (filters) to learn in the temporal convolutional
        layer 2. Higher number increase model capacity but risk overfitting. Deafult 
        is 18.
    - temp_conv_size_sec : float, optional
        The physical size of the temporal filters in seconds. This dictates how long
        of a "time window" the network looks at in one step. Default is 0.5 seconds 
        i.e., 50 samples per window
    - max_pool_size_sec : float, optional
        The size of the max pooling window in seconds, Controls how aggressively the
        time dimnesion is downsampled. Default is 0.08 seconds i.e., 
        int(0.08 x 100) = 8 samples
    - dropout_rate : float, optional
        The probability of dropping a neuron in the final dense layer. Used for 
        regularisation to prevent overfitting. Default to 0.25 (25%).
    """
    def __init__(
        self,
        eeg_epoch_duration : float,
        n_channels : int,
        sfreq : float,
        n_classes : int,
        n_spatial_filters  : int = 8,
        n_temporal_filters_l1 : int = 18,
        n_temporal_filters_l2 : int = 18,
        temp_conv_size_sec : float = 0.5,
        max_pool_size_sec : float = 0.08,
        dropout_rate : float = 0.25
    ):
        super().__init__()

        self.n_channels = n_channels

        # Convert physical sizes (seconds) to tensor sizes (number of samples)
        # e.g., 0.5 seconds x 100 Hz - 50 samples wide kernel
        temp_conv_samples = int(temp_conv_size_sec * sfreq)
        max_pool_samples = int(max_pool_size_sec * sfreq)

        # padding/border is requirerd to ensure the temporal convolutions don't shrink
        # the time dimension around the edges. A padding/border of kernel_size // 2 
        # around the edge, let's the sample point on the edge get filtered as well.
        pad_size = temp_conv_samples // 2


        # ---- Block 1 : Spatial Convolution ----
        # learns optimal linear combinations of the physical EEG electrodes.
        if n_channels > 1: # when only we have more than 1 eeg channel we want to perform Spatial Convolutio
            self.spatial_conv = nn.Sequential(
                # Input shape: (Batch, Dummy_Depth=1, Height/Channels, Width/Time)
                # we want a filter that spans "all channels" but only "1 time step".
                nn.Conv2d(
                    in_channels=1, # 1 input channel (the dummy depth dimension)
                    out_channels=n_spatial_filters, # Number of new "virtual" channels / number of kernels
                    kernel_size=(n_channels, 1) # filter with N channels and 1 sample each channel
                )
            )

        
        # ---- Block 2 : Temporal Feature Extraction ----
        # Finds time-based patterns (spindles, slow waves) within the virtual channels.
        self.temporal_extractor = nn.Sequential(
            # -- First Temporal Layer --
            # Input : (Batch, 1, Virtual Channels, Time)
            nn.Conv2d(
                in_channels=1,
                out_channels=n_temporal_filters_l1,
                kernel_size=(1, temp_conv_samples),
                padding=(0, pad_size) # padding only on the width edges as thier is no height to worry about
            ),
            nn.LeakyReLU(), # Leaky Relu activate function
            nn.MaxPool2d( # Max Pooling
                kernel_size=(1, max_pool_samples)
            ),

            # -- Second Temporal Layer --
            # Input : (Batch, n_temporal_filters, Virtual Channels, Time)
            nn.Conv2d(
                in_channels=n_temporal_filters_l1,
                out_channels=n_temporal_filters_l2,
                kernel_size=(1, temp_conv_samples),
                padding=(0, pad_size)
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=(1, max_pool_samples)
            )
        )


        # ---- Block 3 : Classifier (Fully Connectedd Layer) ----
        # Maps the extracted features to the final 5 sleep stage probabilities.
        # --
        # We must calculate exactly how many features will emerge from the temporal
        # extractor so we can size the final Linear layer correctly.
        # A dummy forward pass is used to compute the exact flattened size, since
        # padding and floor-division in MaxPool make a closed-form formula error-prone.
        n_times = int(eeg_epoch_duration * sfreq)
        dummy = torch.zeros(1, 1, n_channels, n_times)
        if n_channels > 1:
            dummy = self.spatial_conv(dummy)
            dummy = dummy.transpose(1, 2)
        dummy = self.temporal_extractor(dummy)
        flattened_size = dummy.shape[1:].numel()

        self.classifier = nn.Sequential(
            # Unrolls the 3D feature map into a flat 1d vector (keeping batch dim)
            nn.Flatten(start_dim=1),
            # Reguralisation to prevent the network from memorizing the training data
            nn.Dropout(dropout_rate),

            # Final dense layer outputting the 5 class logits
            nn.Linear(
                in_features=flattened_size,
                out_features=n_classes
            ) 
        )


    def forward(self, x : torch.tensor):
        """
        Define the forward pass data flow.

        Parameters
        ----------
        x : torch.tensor
            Batch of EEG windows.
            Expected shape: (batch_size, 1 (dummy_depth), n_channels, n_times)
            - Example for batch size 32 with EEG epoch time window of 30s and 100 Hz
            sampling rate (i.e., 3000 samples) and 2 EEG channels: (32, 1, 2, 3000)

        Returns
        -------
        torch.Tensor
            Logits for the 5 sleep stages.
            Shape: (batch_size, n_classes)
        """
        # Step 1. Spatil Processing
        if self.n_channels > 1:
            x = self.spatial_conv(x)
            # Permute dimensions:
            #   (Batch, Virtual_Filters, 1, Time) -> (Batch, 1, Virtual_Filters, Time)
            # This prepare the tensor for the temporal convolutions.
            x = x.transpose(1, 2)

        # Step 2. Temporal Processing & Downsampling
        x = self.temporal_extractor(x)

        # Final step: Flatten and Classify
        out = self.classifier(x)

        return out