# Fast gradient sign method

An Proof of Concept implementing the method for the creation of adversarial examples proposed in the [EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES](https://arxiv.org/pdf/1412.6572.pdf)

## Installation

```
# Creation of virtual environment is not mandatory, but recommended.
$ python3 -m venv env
$ . env/bin/activate

# proper installation
$ pip install -r requirements.txt
```

## Usage
For automatic example of the implemenation of the attack.

```
$ python poc.py
```

For a more custom test of the model results you can use

```
$ python classifier.py 
Usage: python classifier.py <target_image>
Example: python classifier.py giant_panda.jpg
```
An adversarial example of the proposed image can be tested:

```
$ python classifier.py giant_panda.jpg
$ python classifier.py adversarial_example.jpg
```

## Results
![](./Docs/results.png)
