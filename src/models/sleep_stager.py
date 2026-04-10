"""
Sleep stage classification CNN architecture (Chambon et al. 2018 variant).

This module defines the SleepStager class, which is a PyTorch based CNN
architecture implementation used for sleep staging tasks (Chambon et al. 2018 variant).
However, in this model instead of using ReLU to introduce non-linearity we use 
LeakyRelu to prevent dying neurons with the use of a leak, and we have also added batch
normalization layers after each convolutional layer to stabilize and speed up training
as it helps to reduce internal covariate shift.


References
----------
Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., & Gramfort, A. (2018).
A deep learning architecture for temporal sleep stage classification using
multivariate and multimodal time series. IEEE Transactions on Neural Systems
and Rehabilitation Engineering, 26(4), 758-769.


##### Author: Kartik M. Jalal
##### Last Updated: 04-04-2026
"""
import torch
from torch import nn

class SleepStager(nn.Module):
    """
    PyTorch implementation of the sleep staging architecture.

    This Convolutional Neural Network (ConvNet) is designed specifically for
    multivariate EEG and EOG time-series data. It uses a two-step feature extraction
    process:
        1. Spatial Convolutions: Extracts features across different physical 
            electrodes.
        2. Temporal Convolutions: Extracts time-based features (like sleep spindles)
            from the virtual spatial channels.
    

    Parameters
    ----------
    - signal_epoch_duration : float
        The duration of each labeled digital signal data epoch
    - n_channels : int
        The number of physical channels in the digital signal data, e.g., 3 for 
        FPZ-Cz EEG, Pz-Oz EEG and EOG channels.
    - sfreq : float
        The sampling frequency of the digital signal data in Hz (e.g., a sampling rate 
        of 100.0 Hz means a sample is recorded every 0.01 seconds or 
        0.01 x 1000 = 10 milliseconds).
    - n_classes : int
        The number of sleep stages to predict (e.g., 5 for W, N1, N2, N3, R).
    - n_spatial_filters : int, optional
        The number of feature maps (filters) to learn in the spatial convolutional
        layers. Higher number increase model capacity but risk overfitting. Default
        is 6.
    - n_temporal_filters_l1 : int, optional
        The number of feature maps (filters) to learn in the temporal convolutional
        layer 1. Higher number increase model capacity but risk overfitting. Default
        is 12.
    - n_temporal_filters_l2 : int, optional
        The number of feature maps (filters) to learn in the temporal convolutional
        layer 2. Higher number increase model capacity but risk overfitting. Default
        is 18.
    - temp_conv_size_sec : float, optional
        The physical size of the temporal filters in seconds. This dictates how long
        of a "time window" the network looks at in one step. Default is 0.5 seconds
        i.e., 50 samples per window at 100 Hz.
    - max_pool_size_sec : float, optional
        The size of the max pooling window in seconds. Controls how aggressively the
        time dimension is downsampled. Default is 0.25 seconds i.e.,
        int(0.25 x 100) = 25 samples.
    - dropout_rate : float, optional
        The probability of dropping a neuron in the final dense layer. Used for
        regularisation to prevent overfitting. Default is 0.5 (50%).
    """
    def __init__(
        self,
        signal_epoch_duration : float,
        n_channels : int,
        sfreq : float,
        n_classes : int,
        n_spatial_filters  : int = 6,
        n_temporal_filters_l1 : int = 12,
        n_temporal_filters_l2 : int = 18,
        temp_conv_size_sec : float = 0.5,
        max_pool_size_sec : float = 0.25,
        dropout_rate : float = 0.5
    ):
        super().__init__()

        self.n_channels = n_channels

        # Convert physical sizes (seconds) to tensor sizes (number of samples)
        # e.g., 0.5 seconds x 100 Hz → 50 samples wide kernel
        temp_conv_samples = int(temp_conv_size_sec * sfreq)
        max_pool_samples = int(max_pool_size_sec * sfreq)

        # ---- Block 1 : Spatial Convolution ----
        # Learns optimal linear combinations of the physical EEG and EOG electrodes.
        if n_channels > 1: # only when we have more than 1 channel we want to perform Spatial Convolution
            self.spatial_conv = nn.Sequential(
                # Input shape: (Batch, Dummy_Depth=1, Height/Channels, Width/Time)
                # we want a filter that spans "all channels" but only "1 time step".
                nn.Conv2d(
                    in_channels=1, # 1 input channel (the dummy depth dimension)
                    out_channels=n_spatial_filters, # Number of new "virtual" channels / number of kernels
                    kernel_size=(n_channels, 1) # filter with N channels and 1 sample each channel
                ),
                # Perform batch normalization to stabilize training and improve convergence
                nn.BatchNorm2d(n_spatial_filters)
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
                padding='same' # preserves the time dimension exactly (no off-by-one for even kernels)
            ),
            # Perform batch normalization to stabilize training and improve convergence
            nn.BatchNorm2d(n_temporal_filters_l1),
            # Leaky Relu activate function to introduce non-linearity and prevent dying neurons
            nn.LeakyReLU(),
            # Max Pooling to downsample the time dimension and reduce computational load, while keeping the most salient features.
            nn.MaxPool2d( 
                kernel_size=(1, max_pool_samples)
            ),

            # -- Second Temporal Layer --
            # Input : (Batch, n_temporal_filters, Virtual Channels, Time)
            nn.Conv2d(
                in_channels=n_temporal_filters_l1,
                out_channels=n_temporal_filters_l2,
                kernel_size=(1, temp_conv_samples),
                padding='same'
            ),
            nn.BatchNorm2d(n_temporal_filters_l2),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=(1, max_pool_samples)
            )
        )


        # ---- Block 3 : Classifier (Fully Connected Layer) ----
        # Maps the extracted features to the final sleep stage logits.
        # We must calculate exactly how many features will emerge from the temporal
        # extractor so we can size the final Linear layer correctly.
        # A dummy forward pass is used to compute the exact flattened size, since
        # padding and floor-division in MaxPool make a closed-form formula error-prone.
        n_times = int(signal_epoch_duration * sfreq)
        dummy = torch.zeros(1, 1, n_channels, n_times)
        if n_channels > 1:
            dummy = self.spatial_conv(dummy)
            dummy = dummy.transpose(1, 2)
        dummy = self.temporal_extractor(dummy)
        flattened_size = dummy.shape[1:].numel()

        self.classifier = nn.Sequential(
            # Unrolls the 3D feature map into a flat 1d vector (keeping batch dim)
            nn.Flatten(start_dim=1),
            # Regularisation to prevent the network from memorizing the training data
            nn.Dropout(dropout_rate),

            # Final dense layer outputting the n_classes logits
            nn.Linear(
                in_features=flattened_size,
                out_features=n_classes
            ) 
        )


    def forward(self, x : torch.Tensor):
        """
        Define the forward pass data flow.

        Parameters
        ----------
        x : torch.Tensor
            Batch of EEG and EOG windows.
            Expected shape: (batch_size, 1 (dummy_depth), n_channels, n_times)
            - Example for batch size 32 with digital signal epoch time window of 30s and 100 Hz
            sampling rate (i.e., 3000 samples) and 2 EEG and 1 EOG channels: (32, 1, 3, 3000)

        Returns
        -------
        torch.Tensor
            Logits for the 5 sleep stages.
            Shape: (batch_size, n_classes)
        """
        # Step 1. Spatial Processing
        if self.n_channels > 1:
            x = self.spatial_conv(x)
            # Permute dimensions:
            #   (Batch, Virtual_Filters, 1, Time) -> (Batch, 1, Virtual_Filters, Time)
            # The spatial conv collapses the physical channels (height) to 1, so the
            # output filters sit in dim=1. The temporal conv expects in_channels=1
            # with the virtual channels as "height", so we swap dims 1 and 2.
            # After this, each temporal kernel (1, temp_conv_samples) slides along
            # the time axis independently for each virtual channel row.
            x = x.transpose(1, 2)

        # Step 2. Temporal Processing & Downsampling
        x = self.temporal_extractor(x)

        # Final step: Flatten and Classify
        out = self.classifier(x)

        return out