from cStringIO import StringIO
import os
import re

from simplecv.core.image import image_method


@image_method
def upload(img, dest, api_key=None, api_secret=None, verbose=True):
    """
    **SUMMARY**

    Uploads image to imgur or flickr or dropbox. In verbose mode URL values
    are printed.

    **PARAMETERS**

    * *api_key* - a string of the API key.
    * *api_secret* (required only for flickr and dropbox ) - a string of
     the API secret.
    * *verbose* - If verbose is true all values are printed to the screen


    **RETURNS**

    if uploading is successful

    - Imgur return the original image URL on success and None if it fails.
    - Flick returns True on success, else returns False.
    - dropbox returns True on success.


    **EXAMPLE**

    TO upload image to imgur::

      >>> img = Image("lenna")
      >>> result = upload(img, 'imgur',"MY_API_KEY1234567890" )
      >>> print "Uploaded To: " + result[0]


    To upload image to flickr::

      >>> upload(img, 'flickr','api_key','api_secret')
      >>> # Once the api keys and secret keys are cached.
      >>> upload(img.invert(), 'flickr')


    To upload image to dropbox::

      >>> img.upload('dropbox','api_key','api_secret')
      >>> # Once the api keys and secret keys are cached.
      >>> upload(img.invert(), 'dropbox')


    **NOTES**

    .. Warning::
      This method requires two packages to be installed

      - PyCurl
      - flickr api.
      - dropbox


    .. Warning::
      You must supply your own API key. See here:

      - http://imgur.com/register/api_anon
      - http://www.flickr.com/services/api/misc.api_keys.html
      - https://www.dropbox.com/developers/start/setup#python

    """
    if dest == 'imgur':
        try:
            import pycurl
        except ImportError:
            print "PycURL Library not installed."
            return

        response = StringIO()
        c = pycurl.Curl()
        values = [("key", api_key),
                  ("image", (c.FORM_FILE, img.filename))]
        c.setopt(c.URL, "http://api.imgur.com/2/upload.xml")
        c.setopt(c.HTTPPOST, values)
        c.setopt(c.WRITEFUNCTION, response.write)
        c.perform()
        c.close()

        match = re.search(r'<hash>(\w+).*?<deletehash>(\w+)'
                          r'.*?<original>(http://[\w.]+/[\w.]+)',
                          response.getvalue(), re.DOTALL)
        if match:
            if verbose:
                print "Imgur page: http://imgur.com/" + match.group(1)
                print "Original image: " + match.group(3)
                print "Delete page: http://imgur.com/delete/" \
                      + match.group(2)
            return [match.group(1), match.group(3), match.group(2)]
        else:
            if verbose:
                print "The API Key given is not valid"
            return None

    elif dest == 'flickr':
        global temp_token
        flickr = None
        try:
            import flickrapi
        except ImportError:
            print "Flickr API is not installed. Please install it from " \
                  "http://pypi.python.org/pypi/flickrapi"
            return False
        try:
            if not (api_key is None and api_secret is None):
                img.flickr = flickrapi.FlickrAPI(api_key, api_secret,
                                                 cache=True)
                img.flickr.cache = flickrapi.SimpleCache(timeout=3600,
                                                         max_entries=200)
                img.flickr.authenticate_console('write')
                temp_token = (api_key, api_secret)
            else:
                try:
                    img.flickr = flickrapi.FlickrAPI(temp_token[0],
                                                     temp_token[1],
                                                     cache=True)
                    img.flickr.authenticate_console('write')
                except NameError:
                    print "API key and Secret key are not set."
                    return
        except:
            print "The API Key and Secret Key are not valid"
            return False
        if img.filename:
            try:
                img.flickr.upload(img.filename, img.filehandle)
            except:
                print "Uploading Failed !"
                return False
        else:
            tf = img.save(temp=True)
            img.flickr.upload(tf, "Image")
        return True

    elif dest == 'dropbox':
        global dropbox_token
        access_type = 'dropbox'
        try:
            from dropbox import client, rest, session
            import webbrowser
        except ImportError:
            print "Dropbox API is not installed. For more info refer : " \
                  "https://www.dropbox.com/developers/start/setup#python "
            return False
        try:
            if 'dropbox_token' not in globals() \
                    and api_key is not None and api_secret is not None:
                sess = session.DropboxSession(api_key, api_secret,
                                              access_type)
                request_token = sess.obtain_request_token()
                url = sess.build_authorize_url(request_token)
                webbrowser.open(url)
                print "Please visit this website and press the 'Allow' " \
                      "button, then hit 'Enter' here."
                raw_input()
                access_token = sess.obtain_access_token(request_token)
                dropbox_token = client.DropboxClient(sess)
            else:
                if dropbox_token:
                    pass
                else:
                    return None
        except:
            print "The API Key and Secret Key are not valid"
            return False
        if img.filename:
            try:
                f = open(img.filename)
                dropbox_token.put_file(
                    '/SimpleCVImages/' + os.path.split(img.filename)[-1],
                    f)
            except:
                print "Uploading Failed !"
                return False
        else:
            tf = img.save(temp=True)
            f = open(tf)
            dropbox_token.put_file('/SimpleCVImages/' + 'Image', f)
            return True
