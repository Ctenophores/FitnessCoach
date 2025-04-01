import dataset


def main():
    dataset.split_dataset_files(source_dir='skeleton_data', output_root='split_data', train_ratio=0.8)
    train_loader, val_loader = dataset.get_dataloaders(data_root='split_data', batch_size=8)

    for batch_data, batch_labels in train_loader:
        # print("Data shape:", batch_data.shape)   # (batch_size, T, 132)
        print("Labels:", batch_labels)
        print("Packed data shape:", batch_data.data.shape)  # 展平后的所有有效帧数据
        print("Batch sizes:", batch_data.batch_sizes)
        break  ########only for tets dataloader

if __name__ == '__main__':
    main()