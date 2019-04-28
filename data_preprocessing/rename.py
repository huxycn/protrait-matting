import os


if __name__ == '__main__':
    raw_images_dir = '/home/work/DATA/PortraitMatting/raw/images'
    all_data_urls_file = '/home/work/DATA/PortraitMatting/raw/alldata_urls.txt'

    for img_name in sorted(os.listdir(raw_images_dir)):
        print(img_name)


