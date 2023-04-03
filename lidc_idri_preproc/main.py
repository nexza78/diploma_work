import pylidc as pl
import os
import matplotlib.pyplot as plt
'''
ann = pl.query(pl.Annotation).first()
print(ann)
i, j, k = ann.centroid

vol = ann.scan.to_volume()

plt.imshow(vol[:, :, int(k)], cmap=plt.cm.gray)
plt.plot(j, i, '.r', label="Nodule centroid")
plt.legend()
plt.show()'''
'''
scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == "LIDC-IDRI-0001")
print(scans.count())

ann = pl.query(pl.Annotation).first()
print(ann)

ann = pl.query(pl.Annotation).filter(pl.Scan.patient_id == "LIDC-IDRI-0001").first()

vol = ann.scan.to_volume()

padding = [(30,10), (10,25), (0,0)]

mask = ann.boolean_mask(pad=padding)
bbox = ann.bbox(pad=padding)

fig,ax = plt.subplots(1,2,figsize=(5,3))

ax[0].imshow(vol[bbox][:,:,2], cmap=plt.cm.gray)
ax[0].axis('off')

ax[1].imshow(mask[:,:,2], cmap=plt.cm.gray)
ax[1].axis('off')

plt.tight_layout()
#plt.savefig("../images/mask_bbox.png", bbox_inches="tight")
plt.show()'''

pid = 'LIDC-IDRI-0001'
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
test = []

ann = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
nods = ann.cluster_annotations()
print(len(ann.annotations))
for i, nod in enumerate(nods):
    print("nodule", i, "have ", len(nod), "annotations")
'''
for i in range(1019):
    pid = 'LIDC-IDRI-' + str(0)*(4-len(str(i))) + str(i)
    print(pid)
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
    test.append(len(scan))
vol = scan.to_volume()
print(vol.shape) # (dim, dim, depth)

plt.figure(figsize=(5, 5))
#for i in range(vol.shape[2]):
plt.imshow(vol[:,:,0])
plt.show()

import numpy as np

w = 100
h = 100
fig = plt.figure(figsize=(100, 100))
columns = 5
rows = 5
#for i in range(1, columns*rows+1):
for i in range(1, columns*rows+1):
    #img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(vol[:, :, i-1], cmap='gray')
    # for i in range(vol.shape[2]):
plt.show()'''