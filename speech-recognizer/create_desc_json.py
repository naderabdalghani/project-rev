""" Utility script to convert common voice into wav and create the training, validation and test json files for.
"""
import argparse
import json
import random
import csv
from pydub import AudioSegment
from mutagen.mp3 import MP3

FILE_PATH = '../../Common Voice Dataset/validate.tsv'  # Contains file path of validate.tsv
JSON_PATH = '../../Common Voice Dataset'  # Contains directory for json files

def main(args):
    data = []
    directory = args.file_path.rpartition('/')[0]
    percent = args.percent

    with open(args.file_path, encoding="utf8") as f:
        length = sum(1 for line in f)

    with open(args.file_path, newline='', encoding="utf8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        index = 1
        if args.convert:
            print(str(length) + " files found")
        for row in reader:
            file_name_mp3 = row['path']
            file_name_wav = file_name_mp3.rpartition('.')[0] + ".wav"
            text = row['sentence']
            audio = MP3(directory + "/clips/" + file_name_mp3)
            duration = audio.info.length
            if args.convert:
                data.append({
                    "key": directory + "/clips/" + file_name_wav,
                    "duration": duration,
                    "text": text
                })
                print("converting file " + str(index) + "/" + str(length) + " to wav", end="\r")
                src = directory + "/clips/" + file_name_mp3
                dst = directory + "/clips/" + file_name_wav
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
                index = index + 1
            else:
                data.append({
                    "key": directory + "/clips/" + file_name_mp3,
                    "duration": duration,
                    "text": text
                })

    random.shuffle(data)
    print("creating JSON's")
    f = open(args.save_json_path + "/" + "train.json", "w")

    with open(args.save_json_path + "/" + 'train.json', 'w') as f:
        d = len(data)
        i = 0
        while i < int(d - d / percent):
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1

    f = open(args.save_json_path + "/" + "test.json", "w")

    with open(args.save_json_path + "/" + 'test.json', 'w') as f:
        d = len(data)
        i = int(d - d / percent)
        while i < d:
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to convert common voice into wav and create the training and test json files. """
                                     )
    parser.add_argument('--file_path', type=str, default=FILE_PATH,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, default=JSON_PATH,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')

    args = parser.parse_args()

    main(args)
