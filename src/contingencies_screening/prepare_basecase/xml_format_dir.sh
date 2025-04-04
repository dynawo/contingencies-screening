#!/bin/bash
#
#
# xml_format_dir.sh: given a base directory containing any Dynawo,
# Astre, or Hades cases, cases, format all XML files using
# xmllint. This is useful so that we can later parse with lxml
# ignoring whitespace and always obtaining the same indentation (for
# the benefit of obtaining clean diffs!).
#
# (c) Grupo AIA
# marinjl@aia.es
#
#

# For saner programming:
set -o nounset -o noclobber
set -o errexit -o pipefail 


usage()
{
    cat <<EOF

Usage: $0 XMLCASEDIR

  Example: $0 PtFige-Lille/20190410_1200
    (will format all xml files and leave them under PtFige-Lille/20190410_1200.FORMATTED) 

  NOTE: you can control the indentation with the environment variable XMLLINT_INDENT.
  Example: XMLLINT_INDENT="   " $0 PtFige-Lille/20190410_1200
     (will indent using three spaces, instead of two which is the default)

EOF
}



if [ $# -ne 1 ]; then
    usage
    exit -1
fi
ORIG=${1%/}  # remove possible trailing slash

DEST="$ORIG.FORMATTED"
rm -rf "$DEST"
cp -a "$ORIG" "$DEST"

find "$DEST" -type f | while read -r FILE; do
    if (file -b "$FILE" | grep -qiw "XML"); then
        echo -ne "Formatting: $FILE\\t\\t\\t... "
        xmllint --format "$FILE" > "$FILE.FMT"
        mv "$FILE.FMT" "$FILE"
        echo "(OK)"
    fi
done

