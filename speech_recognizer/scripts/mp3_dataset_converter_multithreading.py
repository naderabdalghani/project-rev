""" Utility script to convert common voice into wav and create the training, validation and testing json files for.
    This file is run once only to get the converted data
"""
import threading
import json
import csv
from tqdm import tqdm
from pydub import AudioSegment
from mutagen.mp3 import MP3
from ..config import COMMON_VOICE_TSV_FILE_PATH, CREATED_JSON_PATH, NUM_OF_THREADS


def main(file_path=COMMON_VOICE_TSV_FILE_PATH, valid_percent=10, test_percent=10, save_json_path=CREATED_JSON_PATH):
    global data
    data = []
    valid_percent = valid_percent
    test_percent = test_percent
    lock = threading.Lock()
    with open(file_path, encoding="utf8") as f:
        length = sum(1 for line in f)

    with open(file_path, newline='', encoding="utf8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        reader_list = list(reader)
        if convert:
            print(str(length) + " files found")
        thread_len = length // NUM_OF_THREADS
        threads = []
        # Create multiple threads with number equals to NUM_OF_THREADS
        for t in range(NUM_OF_THREADS):
            start_idx = t * thread_len
            end_idx = (t+1) * thread_len
            if t == NUM_OF_THREADS - 1:
                end_idx = length
            threads.append(threading.Thread(target=convert, name="Thread#"+str(t),args=(lock, reader_list[start_idx:end_idx],)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

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


def convert(lock, reader_list, convert=True, file_path=COMMON_VOICE_TSV_FILE_PATH):
    global data
    directory = file_path.rpartition('/')[0]
    index = 1
    for row in tqdm(reader_list, desc=threading.currentThread().name):
        file_name_mp3 = row['path']
        file_name_wav = file_name_mp3.rpartition('.')[0] + ".wav"
        text = row['sentence']
        audio = MP3(directory + "/clips/" + file_name_mp3)
        duration = audio.info.length
        if convert:
            # Lock data before writing in it
            lock.acquire()
            data.append({
                "key": directory + "/clips/" + file_name_wav,
                "duration": duration,
                "text": text
            })
            lock.release()
            # Release after appending
            src = directory + "/clips/" + file_name_mp3
            dst = directory + "/clips_wav/" + file_name_wav
            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")
            index = index + 1
        else:
            lock.acquire()
            data.append({
                "key": directory + "/clips/" + file_name_mp3,
                "duration": duration,
                "text": text
            })
            lock.release()
    print(threading.currentThread().name + ": Converting Done!")


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="""
    Utility script to convert common voice into wav and create the training, validation and test json files. """
                                     )
    parser.add_argument('--file_path', type=str, default=COMMON_VOICE_TSV_FILE_PATH,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, default=CREATED_JSON_PATH,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--valid_percent', type=int, default=10, required=False,
                        help='percent of clips put into valid.json instead of train.json')
    parser.add_argument('--test_percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')

    args = parser.parse_args()
    '''

    main()
