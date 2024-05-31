from torch.utils.data import Dataset
import torch
import pickle


class QM9Dataset(Dataset):
    def __init__(self, path: str, device: str='cpu') -> None:
        super().__init__()
        self.path = path
        self.raw_data = pickle.load(open(path, "rb"))
        self.device = device
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        elements = torch.tensor(self.raw_data[idx][0], dtype=torch.long, device=self.device)
        positions = torch.tensor(self.raw_data[idx][1], dtype=torch.float32, device=self.device)
        positions -= positions.mean(dim=0) # zero CoM
        return elements, positions 
    
    @staticmethod
    def collate_fn(batch):
        output = {}
        elements, positions = zip(*batch)
        output['atomic_numbers'] = torch.cat(elements, dim=0)
        output['pos'] = torch.cat(positions, dim=0)
        output['mask'] = torch.tensor(
            [mol_size for mol_size in map(len, elements)],
            dtype=torch.long,
            device=output['atomic_numbers'].device
        )
        return output
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = QM9Dataset("data/qm9/train.pkl", device)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
    for batch in loader:
        print(batch)
        assert len(batch['atomic_numbers']) == torch.sum(batch['mask'])\
            and torch.sum(batch['mask']) == len(batch['pos'])
        break
    
