#!/bin/bash

cd /home/step305/Door/AddUser/

#python3 add_new_user.py

id="User"
n=1
# Increment $N as long as a directory with that name exists
while [[ -d "/home/step305/Door/AddUser/KnownFaces/${id}${n}" ]] ; do
    n=$(($n+1))
done

mkdir "/home/step305/Door/AddUser/KnownFaces/${id}${n}"

cp "/home/step305/Door/AddUser/NewUser/"* "/home/step305/Door/AddUser/KnownFaces/${id}${n}/"
cp "/home/step305/Door/AddUser/cardID.txt" "/home/step305/Door/AddUser/KnownFaces/${id}${n}/cardID.txt"
rm "/home/step305/Door/AddUser/NewUser/"*

python3 face_train.py

rm -rf "/home/step305/Door/KnownFaces"

cp -r "/home/step305/Door/AddUser/KnownFaces" "/home/step305/Door/"

rm "/home/step305/Door/trained_knn_model.clf"

cp "/home/step305/Door/AddUser/trained_knn_model.clf" "/home/step305/Door/trained_knn_model.clf"
