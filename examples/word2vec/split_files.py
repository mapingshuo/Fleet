import os
import argparse


def params_args(args=None):
    params = argparse.ArgumentParser(description='split file')
    params.add_argument("--split_part", type=int, default=1024)
    params.add_argument("--output_path", type=str, default="")
    params.add_argument("--file_path", type=str, default="")
    params = params.parse_args()
    return params


def get_example_num(file_list):
    if len(file_list) == 0:
        raise ValueError("File list Path is empty, please check your path")
    count = 0
    for f in file_list:
        last_count = count
        for index, line in enumerate(open(f, 'r')):
            count += 1
        print("file: %s has %s examples" % (f, count-last_count))
    print("Total example: %s" % count)
    return count


def split_file(file_list, split_part, output_path):
    total_examples = get_example_num(file_list)
    if total_examples < split_part:
        raise ValueError("Total examples: %s < split part: %s, please check your file or setting" % (
            total_examples, split_part))
    size = total_examples / split_part
    part_size = []
    for i in range(split_part):
        part_size.append(size + (i < total_examples % split_part))
    curr_part = 0
    curr_part_line = []
    count = 0
    for f_r in file_list:
        for index, line in enumerate(open(f_r, 'r')):
            count += 1
            curr_part_line.append(line)
            if count == part_size[curr_part]:
                with open(str(output_path)+"/part-"+str(curr_part), 'w+') as f_w:
                    f_w.writelines(curr_part_line)
                    count = 0
                    curr_part_line = []
                    print("Part-%s done" % curr_part)
                    curr_part += 1


if __name__ == "__main__":
    params = params_args()
    file_list = []
    for _, _, files in os.walk(params.file_path):
        for file in files:
            file_list.append(params.file_path + "/" + file)
    print("File list: {}".format(file_list))
    if not os.path.isdir(params.output_path):
        os.system('mkdir -p ' + params.output_path)
    split_file(file_list, params.split_part, params.output_path)
    print("Split files done")
