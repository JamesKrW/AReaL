# Licensed under the Apache License, Version 2.0 (the "License").

import uuid
import time
from typing import Optional
import torch

from realhf.api.core import data_api


class EnvDataset(torch.utils.data.Dataset):
    """Environment-driven dataset that doesn't require loading external files.
    
    This dataset is used for RL experiments where prompts are generated 
    dynamically by the environment (e.g., Sokoban), rather than loaded 
    from a static dataset file.
    """
    
    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_size: int = 1000,  # Virtual dataset size for framework compatibility
        **kwargs
    ):
        """Initialize the environment dataset.
        
        Args:
            util: Dataset utility class containing tokenizer, seed, distributed info, etc.
            max_length: Maximum sequence length (unused but required for interface)
            dataset_path: Path to dataset file (unused but required for interface)
            dataset_size: Virtual size of the dataset for framework compatibility
        """
        self._util = util
        self.max_length = max_length or 1024
        self.dataset_size = dataset_size
        
        # Create dummy data for framework compatibility
        # Each sample gets a unique query_id that will be used by the agent
        self.dummy_samples = []
        for i in range(dataset_size):
            # Generate unique QID for each sample
            timestamp = int(time.time() * 1000000)  # microseconds
            uuid_part = uuid.uuid4().hex[:8]
            query_id = f"env_traj_{uuid_part}_{timestamp}_{i}"
            
            self.dummy_samples.append({
                "id": f"env_sample_{i}",
                "query_id": query_id,
                "prompt": "Environment will provide the actual prompt",
            })
    
    def __len__(self):
        """Return the virtual size of the dataset."""
        return self.dataset_size
    
    def __getitem__(self, idx):
        """Return a dummy sample.
        
        Note: This method should not be called in practice for environment-driven
        experiments, as the agent gets prompts directly from the environment.
        """
        sample = self.dummy_samples[idx % len(self.dummy_samples)]
        
        # Create a minimal sequence sample with query_id
        dummy_input_ids = [1, 2, 3]  # Dummy token IDs
        
        return data_api.SequenceSample.from_default(
            ids=[sample["query_id"]],  # Use query_id as the sample ID
            seqlens=[len(dummy_input_ids)],
            data=dict(
                packed_prompts=torch.tensor(dummy_input_ids, dtype=torch.long),
                # Don't include query_id in data, it's already in ids
            ),
        )


# Register the dataset
data_api.register_dataset("env-dataset", EnvDataset) 