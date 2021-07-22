""" Utility script to convert common voice into wav and create the training, validation and testing json files for.
    This file is run once only to get the converted data
"""
import multiprocessing
import json
import csv
from tqdm import tqdm
from pydub import AudioSegment
from mutagen.mp3 import MP3

FILE_PATH = 'data/cv-corpus-6.1-2020-12-11/en/validated.tsv'  # Contains file path of validate.tsv
JSON_PATH = 'data/cv-corpus-6.1-2020-12-11/en'  # Contains directory for json files
NUM_OF_PROCESSES = 30


def main(file_path=FILE_PATH, valid_percent=10, test_percent=10, save_json_path=JSON_PATH):
    data = []
    valid_percent = valid_percent
    test_percent = test_percent
    with open(file_path, encoding="utf8") as f:
        length = sum(1 for line in f)

    with open(file_path, newline='', encoding="utf8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        reader_list = list(reader)
        if convert:
            print(str(length) + " files found")
        argument_len = length // NUM_OF_PROCESSES
        arguments = []

        # Split data to be run over multi processes
        for p in range(NUM_OF_PROCESSES):
            start_idx = p * argument_len
            end_idx = (p + 1) * argument_len
            if p == NUM_OF_PROCESSES - 1:
                end_idx = length
            arguments.append(reader_list[start_idx:end_idx])

        # Run data with process equals to number of processes
        with multiprocessing.Pool(NUM_OF_PROCESSES) as p:
            for result in p.imap_unordered(convert, arguments):
                data += result

    print("-----------------All converting Done!------------------")
    print(str(len(data)) + "Files converted!")
    print("Creating JSON's")
    with open(save_json_path + "/" + 'train.json', 'w') as train_file:
        d = len(data)
        i = 0
        train_end = int(d - ((d * (test_percent // 100)) + (d * (valid_percent // 100))))
        while i < train_end:
            r = data[i]
            line = json.dumps(r)
            train_file.write(line + "\n")
            i = i + 1

    with open(save_json_path + "/" + 'valid.json', 'w') as valid_file:
        d = len(data)
        i = int(d - ((d * (test_percent // 100)) + (d * (valid_percent // 100))))
        valid_end = int(d - (d * (test_percent // 100)))
        while i < valid_end:
            r = data[i]
            line = json.dumps(r)
            valid_file.write(line + "\n")
            i = i + 1

    with open(save_json_path + "/" + 'test.json', 'w') as test_file:
        d = len(data)
        i = int(d - (d * (test_percent // 100)))
        while i < d:
            r = data[i]
            line = json.dumps(r)
            test_file.write(line + "\n")
            i = i + 1
    print("Done!")


def convert(reader_list, convert=True, file_path=FILE_PATH):
    data = []
    directory = file_path.rpartition('/')[0]
    process_name = multiprocessing.current_process().name
    for row in tqdm(reader_list, desc=process_name, leave=True, position=0):
        file_name_mp3 = row['path']
        file_name_wav = file_name_mp3.rpartition('.')[0] + ".wav"
        text = row['sentence']
        audio = MP3(directory + "/clips/" + file_name_mp3)
        duration = audio.info.length
        if convert:
            data.append({
                "key": directory + "/clips/" + file_name_wav,
                "duration": duration,
                "text": text
            })
            src = directory + "/clips/" + file_name_mp3
            dst = directory + "/clips_wav/" + file_name_wav
            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")
        else:
            data.append({
                "key": directory + "/clips/" + file_name_mp3,
                "duration": duration,
                "text": text
            })
    print(multiprocessing.current_process().name + ": Converting Done!")
    return data


if __name__ == "__main__":
    main()
