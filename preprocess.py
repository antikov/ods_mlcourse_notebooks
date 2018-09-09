import argparse
from tqdm import tqdm

def convert(input_name, output_name):
    label = {'javascript':1, 'java':2, 'python':3, 'ruby':4, 'php':5,
             'c++':6, 'c#':7, 'go':8, 'scala':9, 'swift':10}
    
    label_set = set(label)
    corrupt_count = 0
    selected_count = 0
    with open(input_name, 'r') as f_in, open(output_name, 'w') as f_out:
        for line in tqdm(f_in):
            line = line.strip()
            if line.count('\t') != 1:
                corrupt_count += 1
                continue
            text, tags = line.split('\t')

            intersect_tags = set(tags.split()) & label_set
            if len(intersect_tags) != 1:
                continue
            
            tag_index = label[list(intersect_tags)[0]]
            text = text.replace(":","")
            text = text.replace("|","")

            f_out.write("{} | {}\n".format(tag_index, text))
            selected_count += 1
    return selected_count, corrupt_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vowpal wabbit text preprocessor')
    parser.add_argument('input', help='input filename')
    parser.add_argument('output', help='output filename')
    args = parser.parse_args()
    sc, cc = convert(args.input, args.output)
    print("{} lines selected, {} lines corrupted.".format(sc, cc))
