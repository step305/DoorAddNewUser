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
