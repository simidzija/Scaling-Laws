# Standard library
import json

def concat_jsonl(inpath: str, outpath: str) -> None:
    with open(inpath, 'r') as infile, open(outpath, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            text_list = entry['text_list']
            outfile.write('\n'.join(text_list))
            outfile.write('\n')