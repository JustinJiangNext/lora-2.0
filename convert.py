import rawpy
import imageio
from PIL import Image
import os

def convert_dng_to_png(dng_file, png_file):
    with rawpy.imread(dng_file) as raw:
        rgb = raw.postprocess()
        img = Image.fromarray(rgb)
        img = img.resize((512, 512))
        img.save(png_file, format="PNG")

def convertFolder(inputPath, outputPath):
    for file in os.listdir(inputPath):
        convert_dng_to_png(os.path.join(inputPath, file), os.path.join(outputPath, file + ".png"))
        print(file)
    print("""


    """ + inputPath + """



    """)
    

if __name__ == "__main__":
    os.system("mkdir formatted")
    convertFolder("tmpbin/fivek_dataset/raw_photos/HQa1to700/photos", "formatted")
    convertFolder("tmpbin/fivek_dataset/raw_photos/HQa701to1400/photos", "formatted")
    convertFolder("tmpbin/fivek_dataset/raw_photos/HQa1400to2100/photos", "formatted")
    convertFolder("tmpbin/fivek_dataset/raw_photos/HQa2101to2800/photos", "formatted")
    convertFolder("tmpbin/fivek_dataset/raw_photos/HQa2801to3500/photos", "formatted")
    convertFolder("tmpbin/fivek_dataset/raw_photos/HQa3501to4200/photos", "formatted")
    convertFolder("tmpbin/fivek_dataset/raw_photos/HQa4201to5000/photos", "formatted")
