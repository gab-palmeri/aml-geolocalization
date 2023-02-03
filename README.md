# Visual GeoLocalization: mapping images to GPS

This is the project of the course **Advanced Machine Laerning** at the Politecnico di Torino.
All the code is written in Python and the project consists in jupyter notebooks to be run in Google Colab.
Relative paper is available [here](paper/paper.pdf).

## Authors
The authors of this project are:
- Gioè Tiziano S300886 - [Github](https://github.com/tizianogioe8)
- Lo Truglio Samuele S295285 - [Github](https://github.com/slotruglio)
- Palmeri Mario Gabriele S302740 - [Github](https://github.com/gab-palmeri)

## How to run the code
All of our tests have been run in Google Colab, so we suggest to run the code in the same environment. Notebooks are divided in several folders:
- **step_2_and_3**: contains the notebooks for the second and third step of the project
- **step_4**: contains the notebooks for the fourth step of the project, our contributions to the project

### Google Drive
We provide links to download datasets and models without using google drive mount in colab, but sometimes this kind of connection is not stable. Typically, we use to mount drive in order to work without any issue.

# Step 2 and 3
In the following lines there are the tables related to the results of the second and third step of the project. All of this data is also available in the associated paper and you can reproduce the results by running the notebooks in the folder **step_2_and_3**.

## Step2 - Table
*The following rows show for the default model the respective R@1/R@5 on the test sets.*
| sf-xs (test) | Tokyo-xs  | Tokyo-night |
| :----------: | :-------: | :---------: |
|  52.2/66.3   | 69.5/84.8 |  50.5/72.4  |

## Step3 - Table
*The following rows show for each model the respective loss function, s, m, and R@1/R@5 on the test sets.*
| Loss Function       |   s   |   m   | sf-xs (test) | Tokyo-xs  | Tokyo-night |
| :------------------ | :---: | :---: | :----------: | :-------: | :---------: |
| Cosplace CosFace    | 30.0  | 0.40  |  52.2/66.3   | 69.5/84.8 |  50.5/72.4  |
| Alternative CosFace | 64.0  | 0.35  |  48.7/60.6   | 71.1/81.9 |  60.0/67.6  |
| Alternative CosFace | 30.0  | 0.40  |  47.8/63.9   | 68.9/83.5 |  50.5/69.5  |
| ArcFace             | 64.0  | 0.50  |  47.6/61.6   | 69.2/82.9 |  52.4/66.7  |
| SphereFace          | 30.0  | 1.50  |  50.4/64.4   | 70.8/85.7 |  54.3/72.4  |

# Google Drive Links
- Step3 [Output of executions](https://drive.google.com/drive/folders/1hcD2do0d-_KDFi02bJ1C4kH8UJpw_S2x?usp=share_link)
- Domain Shift [Output and useful materials](https://drive.google.com/drive/folders/13cR7GGzdRDuq_BKnql62hUcuXhpqhzCB?usp=share_link)
- GeoWarp [Output and useful materials](https://drive.google.com/drive/folders/1M0NNV4zpSbBSaaSY3lYxqzhdXWgjTXEz?usp=share_link)
- Alternative backbones [Output and useful materials](https://drive.google.com/drive/folders/1CowavTcF1iiAuvi0A8m3cV1yxWnYovXd?usp=share_link)
- Query Pre Post Processing [Output and useful materials](https://drive.google.com/drive/folders/1T5m1MP57tCGZ4LSuNzVy5bo6s91XrNke?usp=share_link)
- ModelSoup [Output and useful materials](https://drive.google.com/drive/folders/1Cn4TmdRmSs3p3CK6bin1h2iWCtpxEQJV?usp=share_link)