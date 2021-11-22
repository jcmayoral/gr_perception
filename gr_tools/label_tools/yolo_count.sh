yolo_count_classes(){
  if [ -n "$1" ]; then
    echo "counting classes from file " $1
  else
    echo "Images file must be  supplied."
    return
  fi

  #a=$(cat /home/jose/media/elsevier/datasets/real_rgb_files/nibio_valid.txt | rev | cut -c 4- | rev )
  b=(0,0,0,0)

  #Cycle of file
  sp='/-\|'
  start=0
  total=$(cat $1 | wc -l )
  while read a;
    do
    #printf '\b%.1s' "$sp" 
    echo -ne $start " of " $total \\r
    #sp=${sp#?}${sp%???}
    ((start++))

    #replace jpg by txt
    i=$(echo $a|rev|cut -c 4- | rev)
    #echo $i$"txt"
    #read and iterate over each file
    while read p;
      do
      index=$(echo $p | cut -c 1-1);
      #sum 1 to index
      let b[index]=b[index]+1 ;
    done < $i$"txt";
  done < $1

  for i in b
  do 
   echo "for " i 
  done
  echo "TOTAL " ${b[@]}
  unset a
  unset b
  unset index
}
