import urllib.request
import os

def setup_voting_dataset():
    """
    Download and convert Congressional Voting Records dataset
    Replaces missing data (?) with 'unknown'
    """
    
    # Step 1: Download the dataset if it doesn't exist
    if not os.path.exists('voting_original.data'):
        print("Downloading dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
        try:
            urllib.request.urlretrieve(url, "voting_original.data")
            print("Downloaded voting_original.data")
        except Exception as e:
            print(f"Error downloading: {e}")
            return
    else:
        print("voting_original.data already exists")
    
    # Step 2: Create meta file
    print("\nCreating voting.meta...")
    
    attributes = [
        'handicapped-infants',
        'water-project-cost-sharing',
        'adoption-of-budget-resolution',
        'physician-fee-freeze',
        'el-salvador-aid',
        'religious-groups-in-schools',
        'anti-satellite-test-ban',
        'aid-to-nicaraguan-contras',
        'mx-missile',
        'immigration',
        'synfuels-corporation-cutback',
        'education-spending',
        'superfund-right-to-sue',
        'crime',
        'duty-free-exports',
        'export-administration-act-south-africa'
    ]
    
    with open('voting.meta', 'w') as f:
        for attr in attributes:
            f.write(f"{attr}:y,n,unknown\n")
        f.write("class:democrat,republican\n")
    
    print("Created voting.meta")
    
    # Step 3: Convert data file
    print("\nConverting voting data...")
    
    total_instances = 0
    instances_with_missing = 0
    
    with open('voting_original.data', 'r') as infile, \
         open('voting.data', 'w') as outfile:
        
        for line in infile:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            total_instances += 1
            
            # Count instances with missing data
            if '?' in line:
                instances_with_missing += 1
            
            # Replace all '?' with 'unknown'
            line = line.replace('?', 'unknown')
            
            # format: attr1,attr2,...,attr16,class
            parts = line.split(',')
            class_label = parts[0]      # First column is the class
            attributes_data = parts[1:]  # Rest are attributes
            
            # Write in format: attributes first, class last
            outfile.write(','.join(attributes_data) + ',' + class_label + '\n')
    
    print(f"Created voting.data")
    print(f"\nStatistics:")
    print(f"  Total instances: {total_instances}")
    print(f"  Instances with missing data: {instances_with_missing}")
    print(f"  Percentage with missing data: {instances_with_missing/total_instances*100:.1f}%")
    
    # Step 4: Create train/test split
    print("\nCreating train/test split (70/30)...")
    create_train_test_split()
    
    # Done!
    print("\n" + "="*60)
    print("Files created:")
    print("  - voting.meta")
    print("  - voting.data")
    print("  - voting.train")
    print("  - voting.test")
    print("="*60)
    
    print("\nYou can now run:")
    print("  python3 main.py --mode kfold --data voting.data --meta voting.meta --k 5")
    print("  python3 main.py --mode evaluate --train voting.train --test voting.test --meta voting.meta")


def create_train_test_split(test_size=0.3):
    """
    Split voting.data into train and test sets
    
    Args:
        test_size: Fraction of data to use for testing (default 0.3 = 30%)
    """
    import random
    
    # Read all data
    with open('voting.data', 'r') as f:
        lines = f.readlines()
    
    # Shuffle for random split
    random.seed(42)  # For reproducibility
    random.shuffle(lines)
    
    # Calculate split point
    split_point = int(len(lines) * (1 - test_size))
    
    # Write training data
    with open('voting.train', 'w') as f:
        f.writelines(lines[:split_point])
    
    # Write test data
    with open('voting.test', 'w') as f:
        f.writelines(lines[split_point:])
    
    print(f"voting.train: {split_point} instances ({(1-test_size)*100:.0f}%)")
    print(f"voting.test: {len(lines) - split_point} instances ({test_size*100:.0f}%)")


if __name__ == "__main__":
    setup_voting_dataset()