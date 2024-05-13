//
// Segmentations
//

'use strict';

var Segmentations = (function() {

	// Variables

    var $previewZoneImage = $(".preview-zone-image")
    var $previewZoneDefault = $(".preview-zone-default")
    var $previewFilename = $(".preview-filename")
	var $dropzone = $(".dropzone");
	var $dropzoneWrapper = $('.dropzone-wrapper');
	var $buttonRemovePreview = $('.remove-preview');


	// Methods

    function readFile(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $previewFilename.html('Nama file: ' + input.files[0].name)
                $dropzoneWrapper.removeClass('dragover');
                $previewZoneImage.attr('src', e.target.result)
                showDropzoneImage()
                $buttonRemovePreview.removeClass('d-none')
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    function reset(e) {
        e.wrap('<form>').closest('form').get(0).reset();
        e.unwrap();
    }

    function showDropzoneImage() {
        $previewZoneImage.removeClass('d-none')
        $previewZoneDefault.addClass('d-none')
        $buttonRemovePreview.removeClass('d-none')
    }

    function showDropzoneDefault() {
        $previewZoneImage.addClass('d-none')
        $previewZoneDefault.removeClass('d-none')
        $buttonRemovePreview.addClass('d-none')
        $previewFilename.empty()
    }

    $dropzone.change(function(){
        readFile(this);
    });
       
    $dropzoneWrapper.on('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).addClass('dragover');
    });
       
    $dropzoneWrapper.on('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('dragover');
    });
       
    $buttonRemovePreview.on('click', function() {
        showDropzoneDefault()
        reset($dropzone);
    });

})();
