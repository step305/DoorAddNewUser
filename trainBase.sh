#!/bin/bash
cd /home/step305/Door/AddUser/

python3 face_train.py

rm -rf "/home/step305/Door/KnownFaces"

cp -r "/home/step305/Door/AddUser/KnownFaces" "/home/step305/Door/"

rm "/home/step305/Door/trained_knn_model.clf"

cp "/home/step305/Door/AddUser/trained_knn_model.clf" "/home/step305/Door/trained_knn_model.clf"
