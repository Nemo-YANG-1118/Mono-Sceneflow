
#ifndef SCENE_CONFIGURATION_H_
#define SCENE_CONFIGURATION_H_


#ifdef _BORDER_NGAP_
#undef _BORDER_NGAP_
#endif
#define _BORDER_NGAP_ 2



#ifdef _PER_BUNDLE_NFRAMES_
#undef _PER_BUNDLE_NFRAMES_
#endif
#define _PER_BUNDLE_NFRAMES_ 3


#ifdef _VS_WIN_64_
#undef _VS_WIN_64_
#endif
#define _VS_WIN_64_ 1 ///for using x64

#ifdef cind
#undef cind
#endif

#if _VS_WIN_64_
typedef long long cind;
#else
typedef int cind;
#endif



#endif
