import pickle
import os
import sys

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_path = os.path.join(base_dir, 'data', 'preprocessing_artifacts.pkl')
    
    if not os.path.exists(artifacts_path):
        print(f"Error: {artifacts_path} not found. Please run scripts/13-preprocess.py first.")
        return

    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    classes = artifacts.get('target_classes', [])
    
    if not classes:
        print("Error: No target classes found in the artifacts file.")
        return

    print("=" * 40)
    print("WeatherChart: Genre Class Lookup")
    print("=" * 40)
    print(f"Total classes found: {len(classes)}")
    
    # If arguments are provided (e.g. python check_classes.py pop)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip().lower()
        
        # Check if it's an index
        if query.isdigit():
            idx = int(query)
            if 0 <= idx < len(classes):
                print(f"\n[Index {idx}] -> {classes[idx]}")
            else:
                print(f"\nIndex {idx} is out of range.")
        else:
            # Check if it's a name (partial match)
            found = False
            for i, name in enumerate(classes):
                if query in name.lower():
                    print(f"[Index {i:2}] -> {name}")
                    found = True
            if not found:
                print(f"\nNo genre matches found for query: '{query}'")
    else:
        # Show all classes
        print("\nFull List of Classes:")
        for i, name in enumerate(classes):
            print(f"{i:2}: {name}")
        
        print("\nUsage Tips:")
        print("1. Run without arguments to see the full list.")
        print("2. Pass an index: 'python scripts/check_classes.py 59'")
        print("3. Pass a name:  'python scripts/check_classes.py rock'")

if __name__ == "__main__":
    main()
