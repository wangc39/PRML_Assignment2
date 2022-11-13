import sys
import os.path
import pandas as pd


if __name__ == "__main__":

    faces = ['att_faces', 'orl_faces']

    for face in faces:
        BASE = "/home/wangcong/Course/PRML/PCA_FDA/ORL"
        BASE_PATH = os.path.join(BASE, face)
        SAVE_BASE = BASE_PATH

        SEPARATOR=";"
        label, count = 0, 0
        df = pd.DataFrame(data=None, columns=['dir', 'label'])
        with open(f'{BASE}/{face}.csv', 'w', encoding='utf-8') as f:
            for dirname, dirnames, filenames in os.walk(BASE_PATH):
                for subdirname in dirnames:
                    subject_path = os.path.join(dirname, subdirname)
                    save_subject_path = os.path.join(SAVE_BASE, subdirname)
                    for filename in os.listdir(subject_path):
                        abs_path = "%s/%s" % (subject_path, filename)
                        save_path = "%s/%s" % (save_subject_path, filename)
                        print("%s%s%d" % (abs_path, SEPARATOR, label))
                        f.write("%s%s%d\n" % (save_path, SEPARATOR, label))
                        count = count + 1
                    label = label + 1
            