# Sample images

This directory holds test photos. Nothing in the pipeline depends on any
particular file here — pass any photo with `--image`.

## Getting a sample

```bash
python scripts/fetch_sample.py "Sundar Pichai"
```

That pulls the lead image of an English Wikipedia article. Those images are
freely licensed (CC / public domain — check each file's page on Wikimedia
Commons for exact terms) and, for public figures, are widely reposted, which
gives the reverse-image stage a realistic chance of finding social-media copies.

## Choosing a good demo image

The pipeline can only find what search engines have actually indexed:

- **Good:** a widely circulated photo of a public figure. Reposted across news
  sites and social platforms, so reverse search has something to find.
- **Also good:** a photo you posted publicly yourself, once it has been indexed.
- **Poor:** a private photo that has never been online. There is nothing to
  match, and the pipeline will correctly report `UNVERIFIED`.

If the image is already public at a URL, pass `--image-url` — engines' by-URL
endpoints are considerably more reliable than their upload flows.

## Licensing

Sample images are **not** committed to this repository. Fetch them locally with
the script above and respect the license on each image's source page.
