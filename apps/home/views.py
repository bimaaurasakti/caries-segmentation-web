# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""


from django import template
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader
from django.urls import reverse

from .helper import *
from .preprocessing import *
from .predicts.caries_type import predict as predict_type
from .predicts.caries_segmentation import predict as predict_segmentation, predictSegmentationFullImage

import base64

# @login_required(login_url="/login/")
def index(request):
    return HttpResponseRedirect(reverse('segmentation-full'))

# @login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))

# @login_required(login_url="/login/")
def segmentationFullImage(request):
    print(request.POST)
    context = {}
    html_template = loader.get_template('home/segmentations.html')
    context['title'] = 'Radiograf Utuh'

    if request.method == "POST":
        image = request.FILES['img_logo']
        context['caries_segmentation_result'] = predictSegmentationFullImage(image)
    else:
        context['caries_segmentation_result'] = ''

    return HttpResponse(html_template.render(context, request))
    
# @login_required(login_url="/login/")
def segmentationCroppedImage(request):
    context = {}
    html_template = loader.get_template('home/segmentations.html')
    context['title'] = 'Radiograf Potongan'

    if request.method == "POST":
        image = request.FILES['img_logo']
        context['caries_type_result'] = predict_type(image) 
        if context['caries_type_result'] != 'Normal' :
            context['caries_segmentation_result'] = predict_segmentation(image)
        else :
            with image.open(mode='rb') as file:
                image_data = file.read()
                context['caries_segmentation_result'] = base64.b64encode(image_data).decode('utf-8')
    else:
        context['caries_type_result'] = ''
        context['caries_segmentation_result'] = ''

    return HttpResponse(html_template.render(context, request))