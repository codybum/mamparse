import argparse
import os
import shutil
from pathlib import Path
import pydicom
from bs4 import BeautifulSoup
from skimage.draw import polygon2mask
from PIL import Image
import numpy as np
import pandas as pd

def shoelace(x_y):
    x_y = np.array(x_y)
    x_y = x_y.reshape(-1,2)

    x = x_y[:,0]
    y = x_y[:,1]

    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)

    return area

def extract_points(file_path):

    coords = dict()

    with open(file_path, 'r') as f:
        data = f.read()

    xml_data = BeautifulSoup(data, "xml")
    annotations = xml_data.find('array').find('dict').find('array').findAll('dict')


    for annotation in annotations:

        save_coord = False
        index = None
        annotation_type = None

        keys = annotation.findAll('key')
        for key in keys:

            if 'IndexInImage' == key.get_text():
                index = int(key.find_next_siblings('integer')[0].get_text())

            if 'Name' == key.get_text():
                annotation_type = key.find_next_siblings('string')[0].get_text()

            if 'NumberOfPoints' == key.get_text():
                number_of_points = int(key.find_next_siblings('integer')[0].get_text())
                if number_of_points >= args.min_coord_size:
                    save_coord = True

            if save_coord and (annotation_type in args.annotation_types):
                if 'Point_px' == key.get_text():
                    coord_array = key.find_next_siblings('array')[0].findAll('string')
                    for coord_str in coord_array:
                        coord_str = coord_str.get_text().replace('(', '').replace(')', '').replace(' ', '').split(',')
                        x = float(coord_str[0])
                        y = float(coord_str[1])
                        coord = [y, x]
                        if index not in coords:
                            coords[index] = []
                        coords[index].append(coord)

                    if shoelace(coords[index]) < args.min_coord_area:
                        del coords[index]

    return coords

def getmask(image,coord):

    mask = polygon2mask(image.shape, np.array(coord))
    mask = mask.astype(np.uint8) * 255
    return mask

def addmask(image,coords):

    combined = None
    for id, coord in coords.items():

        mask = getmask(image,coord)

        if combined is not None:
            combined = combined + mask
        else:
            combined = mask

    return combined

def get_image(dicom_file_path, png_file_path):

    ds = pydicom.dcmread(dicom_file_path)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    width, height = final_image.size
    final_image.save(png_file_path)

    return width, height

def clear_output():

    if os.path.exists(args.image_dataset_path):
        shutil.rmtree(args.image_dataset_path)
    os.makedirs(args.image_dataset_path)

    if os.path.exists(args.mask_dataset_path):
        shutil.rmtree(args.mask_dataset_path)
    os.makedirs(args.mask_dataset_path)


def process_dataset():

    data = []

    clear_output()

    dicom_files = list(Path(args.dicom_dataset_path).rglob("*.dcm"))
    for dicom_file in dicom_files:
        xml_file_name = dicom_file.name.split('_')[0] + '.xml'
        xml_file_path = os.path.join(args.xml_dataset_path,xml_file_name)
        if os.path.isfile(xml_file_path):
            coords = extract_points(xml_file_path)
            if len(coords) > 0:

                png_file_name = dicom_file.stem + '.png'
                png_file_path = os.path.join(args.image_dataset_path,png_file_name)

                mask_file_name = dicom_file.stem + '_mask.png'
                mask_file_path = os.path.join(args.mask_dataset_path, mask_file_name)

                dicom_file_path = dicom_file.resolve()
                width, height = get_image(dicom_file_path, png_file_path)

                image = np.ones((height, width, 3), dtype=np.uint8)
                image = 255 * image

                combined_mask = addmask(image, coords)

                im = Image.fromarray(combined_mask)
                im.save(mask_file_path)

                data.append([png_file_path,mask_file_path])
                print('Processed',dicom_file.name,'with',len(coords),'roi(s)')

    pd.DataFrame(data, columns=['image_path', 'mask_path']).to_csv(args.csv_dataset_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='INbreast Parser')
    parser.add_argument('--dicom_dataset_path', type=str, default='dicom', help='name of project')
    parser.add_argument('--xml_dataset_path', type=str, default='xml', help='name of project')
    parser.add_argument('--image_dataset_path', type=str, default='dataset/image', help='name of project')
    parser.add_argument('--mask_dataset_path', type=str, default='dataset/mask', help='name of project')
    parser.add_argument('--csv_dataset_path', type=str, default='dataset.csv', help='name of project')

    parser.add_argument('--annotation_types',default=['Mass'], nargs='+',help='list of implemented models')
    parser.add_argument('--min_coord_size', type=int, default=3, help='name of project')
    parser.add_argument('--min_coord_area', type=int, default=1000, help='name of project')

    # get args
    args = parser.parse_args()

    process_dataset()
