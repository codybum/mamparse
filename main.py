import numpy
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from skimage.draw import polygon2mask

def extract_points(file_path):

    coord = []

    with open(file_path, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")

    d = Bs_data.find('array')
    d = d.find('dict')
    d = d.find('array')

    b_unique = d.findAll('dict')

    for page in b_unique:
        d = page.findAll('key')
        for dd in d:
            if 'Point_px' == dd.get_text():
                #print(dd)
                ddd = dd.find_next_siblings('array')
                dddd = ddd[0].find('string')
                ddddd = dddd.get_text().replace('(','').replace(')','').replace(' ','').split(',')
                x = float(ddddd[0])
                y = float(ddddd[1])
                c = [y,x]
                coord.append(c)

    return np.array(coord)

if __name__ == '__main__':

    file_path = '20586908.xml'
    #polygon = extract_points(file_path)
    coordinates = ([1000, 1000], [2000,500], [2000, 1500])
    polygon = np.array(coordinates)
    # create a black image
    image = np.ones((3000, 3000, 3), dtype=np.uint8)
    image = 255 * image

    mask = polygon2mask(image.shape, polygon)
    mask = mask.astype(np.uint8) * 255
    #result = ma.masked_array(image, np.invert(mask))
    #print(f'{result.shape=} {np.min(result)=} {np.max(result)=}')
    print(f'{mask.shape=} {np.min(mask)=} {np.max(mask)=}')
    #plt.imshow(result)
    plt.imshow(mask)
    #plt.imshow(image)
    plt.show()
