import csv
def main():
    with open('data1forEx1to4/train1_icu_label.csv') as label, \
        open('data1forEx1to4/test1_icu_label.csv') as tlabel:
        labels = csv.reader(label)
        tlabels = csv.reader(tlabel)

        died = 0
        survived = 0
        for row in labels:
            if row[0] == '0':
                died += 1
            else:
                survived += 1
        total = died + survived
        print(f"Training: died: {died/total}, survived: {survived/total}")

        for row in tlabels:
            if row[0] == '0':
                died += 1
            else:
                survived += 1
        total = died + survived
        print(f"Test: died: {died/total}, survived: {survived/total}")
        

if __name__=="__main__":
    main()