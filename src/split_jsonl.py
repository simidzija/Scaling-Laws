# Standard library
import json

def split_jsonl(jsonl_path: str, 
                 train_path: str, 
                 test_path: str,
                 test_size: int) -> None:
    with open(jsonl_path, 'r') as infile:
        total_lines = sum(1 for _ in infile)
        cutoff = total_lines - test_size
        infile.seek(0)

        # Write training data
        with open(train_path, 'w') as outfile:
            for i, line in enumerate(infile):
                if i >= cutoff:
                    break
                entry = json.loads(line)
                text_list = entry['text_list']
                outfile.write('\n'.join(text_list))
                outfile.write('\n')

        # Write test data
        with open(test_path, 'w') as outfile:
            for line in infile:
                entry = json.loads(line)
                text_list = entry['text_list']
                outfile.write('\n'.join(text_list))
                outfile.write('\n')