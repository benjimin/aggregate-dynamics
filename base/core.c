//
//
//
//
//
//
//

inline unsigned char prefilter(unsigned char x) {
    return x & 4;
}
inline char wet(unsigned char x) {
    return prefilter(x) == 128
}
inline char clear(unsigned char x) {
    return wet(x) || (prefilter(x) == 0)
}

inline int first(unsigned char *ptr, int size) {
    int i = 0;
    while (! clear(ptr[i]))
        i++;
    return i;
}

void function(unsigned char *array, int size) {

    for (int i = 0; i < rows; i++) {

    }
}