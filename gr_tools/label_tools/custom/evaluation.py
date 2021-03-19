#!/usr/bin/python
from benchmark.benchmark import read_docs
from custom.image_processing import ImageProcessing

if __name__ == "__main__":
    print "THIS IS NOT GOING TO BE IMPLEMENTED... using pytorch instead"
    if len(sys.argv) <3:
        print "error"
        sys.exit()
    rootpath = sys.argv[1]
    train_filepath = os.path.join(rootpath, "files_train.txt")
    valid_filepath = os.path.join(rootpath, "files_valid.txt")


    proc = ImageProcessing(matches, depth_camera_info)


    X_train, y_train = read_docs(train_filepath)
    X_valid, y_valid = read_docs(valid_filepath)


    if os.path.exists(filepath):
        print "path exists", filepath
        images = open(filepath,'r').readlines()
        print "number of files {}".format(len(images))
        with tqdm(total=len(images)) as pbar:
            for img_filename in images:
                label_filename = img_filename.replace(".jpg", ".txt").rstrip()
                labels = []
                if not os.path.exists(label_filename):
                    print "label file {} not exists".format(label_filename)
                    continue

                with open(label_filename, "r") as fl:
                    labels = [data.strip().split(" ") for data in fl]#)
                fl.close()
                #print (label)
                img = cv2.imread(img_filename.rstrip().replace("txt", "jpg"))#, cv2.IMREAD_GRAYSCALE)

                for l in labels:
                    detections = [d for d in l]
                    detections[1:] = [float(s) for s in detections[1:]]
                    plot_bbs(img, detections, visualize = True, out=out)
                    if detections[0] == "ERROR":
                     counter[-1] += 1
                     continue
                    print float(detections[0])
                    cl_ = int(float(detections[0]))
                    if cl_ < 0 or  cl_ > 3:
                        print "ERRROR.....",cl_, img_filename
                        continue
                    counter[cl_] +=1
                pbar.update(1)
                #print("NEXT")
        #f.close()
    out.release()
    print "FINAL COUNTER", counter
