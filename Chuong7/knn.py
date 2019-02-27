import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Duong dan den tep dataset')
ap.add_argument('-n', '--neighbors', required=False, type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', required=False, type=int, default=-1,
                help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

# Lay danh sach hinh anh
image_paths = list(paths.list_images(args['dataset']))

# Khởi tạo SimplePre Processor và SimpleDatasetLoader và tải dữ liệu
print('[INFO]: Dang load hinh anh....')
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)

# Reshape from (3000, 32, 32, 3) to (3000, 32*32*3=3072)
data = data.reshape((data.shape[0], 3072))

# In thong tin ve muc tieu thu bo nho
print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes / 1024*1000.0)))

# Ma hoa lable duoi dang so nguyen
le = LabelEncoder()
labels = le.fit_transform(labels)

# Chia du lieu training (75%) va du lieu thu (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Huấn luyện và đánh giá trình phân loại k-NN về cường độ pixel thô
print('[INFO]: Qua trinh phan loai bat dau...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])
model.fit(train_x, train_y)
print(classification_report(test_y, model.predict(test_x),
                            target_names=le.classes_))