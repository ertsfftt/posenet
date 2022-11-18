mkdir -p build
cd build || exit

function make_plugin() {
    if ! cmake ..;
    then
      echo "cmake failed."
      return 1
    fi

    if ! (make);
    then
      echo "make failed."
      return 1
    fi

    return 0
}

if make_plugin;
then
  echo "INFO: Build successfully."
else
  echo "ERROR: Build failed."
fi

cd - || exit
