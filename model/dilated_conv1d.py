import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RecursiveLayer (No changes needed, it's generic) ---
class RecursiveLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rate = dilation_rate

        self.W1 = nn.Linear(in_channels, out_channels, bias=False)
        self.W2 = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.activation = nn.ReLU()

    def forward(self, C_t_minus_dl_l_minus_1, C_t_l_minus_1):
        term1 = self.W1(C_t_minus_dl_l_minus_1)
        term2 = self.W2(C_t_l_minus_1)
        sum_terms = term1 + term2 + self.bias
        return self.activation(sum_terms)

# --- DeepRecursiveModel (Adjusted for channel sizes) ---
class DeepRecursiveModel(nn.Module):
    def __init__(self, input_feature_dim, first_layer_out_channels, subsequent_layer_channels, num_layers, dilation_rates):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.first_layer_out_channels = first_layer_out_channels      # C(t,1) channels
        self.subsequent_layer_channels = subsequent_layer_channels    # C(t,l) channels for l > 1
        self.num_layers = num_layers
        self.dilation_rates = dilation_rates

        if len(dilation_rates) != num_layers:
            raise ValueError("Number of dilation rates must match num_layers")

        self.layers = nn.ModuleList()

        # Layer 1: Takes input_feature_dim (150) -> outputs first_layer_out_channels (50)
        self.layers.append(RecursiveLayer(input_feature_dim, first_layer_out_channels, dilation_rates[0]))

        # Layers 2 to num_layers: Take subsequent_layer_channels (50) -> outputs subsequent_layer_channels (50)
        for i in range(1, num_layers):
            self.layers.append(RecursiveLayer(subsequent_layer_channels, subsequent_layer_channels, dilation_rates[i]))

    def forward(self, initial_input_sequence):
        # initial_input_sequence: (batch_size, input_feature_dim, sequence_length)
        
        batch_size, _, sequence_length = initial_input_sequence.shape
        device = initial_input_sequence.device
        
        buffer = [None] * (self.num_layers + 1)
        buffer[0] = initial_input_sequence.transpose(1, 2) # (N, L, C_in_0 = 150)

        for l in range(1, self.num_layers + 1):
            current_layer_module = self.layers[l-1]
            current_dilation_rate = self.dilation_rates[l-1]
            
            # Determine output channels for the current layer's buffer entry
            if l == 1:
                current_layer_out_channels = self.first_layer_out_channels
            else:
                current_layer_out_channels = self.subsequent_layer_channels

            C_tl_sequence = torch.zeros(
                batch_size, sequence_length, current_layer_out_channels, device=device
            )

            for t in range(sequence_length):
                C_t_l_minus_1 = buffer[l-1][:, t, :]

                if t - current_dilation_rate >= 0:
                    C_t_minus_dl_l_minus_1 = buffer[l-1][:, t - current_dilation_rate, :]
                else:
                    C_t_minus_dl_l_minus_1 = torch.zeros(
                        batch_size, buffer[l-1].shape[2], device=device
                    )
                
                C_tl = current_layer_module(C_t_minus_dl_l_minus_1, C_t_l_minus_1)
                C_tl_sequence[:, t, :] = C_tl

            buffer[l] = C_tl_sequence
        
        return buffer[self.num_layers].transpose(1, 2)

# --- Example Usage (Updated for your specific training config) ---
if __name__ == '__main__':
    # Input Data Specifications
    input_feature_dim_train = 150   # 3 channels * 25 joints * 2 persons
    batch_size_train = 8            # Example batch size for training
    sequence_length_train = 255     # Frames per sample in a batch

    # Model Configuration based on your requirements
    # Layer 1 output channels: 50
    first_layer_out_channels_train = 50
    # Layers 2 to 14 input/output channels: 50
    subsequent_layer_channels_train = 50
    
    num_layers_train = 14
    
    # Dilation rates for each layer. You mentioned 14 layers.
    # The sum of (dilation_rate * (kernel_size-1)) + 1 for each layer
    # roughly determines the overall receptive field.
    # With kernel_size=2 and dilation `dl`, the receptive field is `dl+1`
    # You mentioned perception window scale from 2 to 255.
    # Let's assume a sample set of dilation rates that roughly match this.
    # e.g., doubling dilations for the first 7 layers, then repeating:
    dilation_rates_train = [1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64] # This sums to 255+ for receptive field
    if len(dilation_rates_train) != num_layers_train:
        raise ValueError(f"Dilation rates list length ({len(dilation_rates_train)}) must match num_layers ({num_layers_train})")

    print("--- Model Initialization for Training ---")
    model_train = DeepRecursiveModel(
        input_feature_dim=input_feature_dim_train,
        first_layer_out_channels=first_layer_out_channels_train,
        subsequent_layer_channels=subsequent_layer_channels_train,
        num_layers=num_layers_train,
        dilation_rates=dilation_rates_train
    )
    print(model_train)
    
    # Create a dummy batch input tensor matching the specified dimensions
    # (batch_number, 150, 255)
    input_batch_ex = torch.randn(
        batch_size_train, input_feature_dim_train, sequence_length_train
    )
    
    print(f"\nTraining Batch Input (C(t,0)) shape: {input_batch_ex.shape} (N, C_in, L)")

    # Perform a forward pass for training
    print("\n--- Training Forward Pass ---")
    output_C_final_train = model_train(input_batch_ex)

    # The final output should have `subsequent_layer_channels` as its channel dimension
    print(f"Final Output (C(t, {num_layers_train})) shape: {output_C_final_train.shape} (N, C_out, L)")

    expected_output_shape = (batch_size_train, subsequent_layer_channels_train, sequence_length_train)
    assert output_C_final_train.shape == expected_output_shape
    print(f"\nOutput shape matches expected dimensions: {expected_output_shape}.")

    # --- Training Loop Placeholder ---
    # optimizer = torch.optim.Adam(model_train.parameters(), lr=0.001)
    # criterion = nn.MSELoss() # Or whatever loss is appropriate for your task

    # for epoch in range(num_epochs):
    #     # In a real scenario, you'd use a DataLoader to get batches from your 4M frames
    #     # For simplicity, using the dummy batch here
    #     optimizer.zero_grad()
    #     output = model_train(input_batch_ex)
    #     
    #     # Assuming you have a target tensor (e.g., for predicting something at the end of the sequence)
    #     # target = ...
    #     # loss = criterion(output, target) 
    #     # loss.backward()
    #     # optimizer.step()
    #     # print(f"Epoch {epoch+1}, Loss: {loss.item()}")