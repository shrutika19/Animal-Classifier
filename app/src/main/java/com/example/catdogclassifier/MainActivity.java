package com.example.catdogclassifier;

import static android.graphics.Color.parseColor;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.FileUtils;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.charts.HorizontalBarChart;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter;

import org.checkerframework.checker.units.qual.C;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedTransferQueue;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Button classifyBtn;
    TextView prediction;
    protected Interpreter tflite;
    private MappedByteBuffer tfliteModel;
    private TensorImage inputImageBuffer;
    private int imageSizeX;
    private int imageSizeY;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabiltyProcessor;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    private Bitmap bitmap;
    private List<String> labels;
    private HorizontalBarChart mbarChart;
    Uri imageuri;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.imageView);
        classifyBtn = (Button) findViewById(R.id.classifyButton);
        prediction = (TextView) findViewById(R.id.predictionText);

        //importing image from gallery
        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"),12);
            }
        });

        //define the interpreter with tflite model
        try {
            tflite = new Interpreter(loadmodelfile(MainActivity.this));
        } catch (IOException e) {
            e.printStackTrace();
        }

        classifyBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int imagetensorIndex = 0 ;
                int[] imageShape = tflite.getInputTensor(imagetensorIndex).shape();
                imageSizeX = imageShape[1];
                imageSizeY = imageShape[2];
                DataType imagDataType = tflite.getInputTensor(imagetensorIndex).dataType();

                int probabilityTensorIndex = 0;
                int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
                DataType probabilityDataType = tflite.getInputTensor(probabilityTensorIndex).dataType();

                inputImageBuffer = new TensorImage(imagDataType);
                outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
                probabiltyProcessor = new TensorProcessor.Builder().add(getPostProcessorNormalizeOP()).build();

                inputImageBuffer = loadImage(bitmap);

                tflite.run(inputImageBuffer.getBuffer(),outputProbabilityBuffer.getBuffer().rewind());
                showResults();
            }
        });

    }

    //load the image and image processing
    private TensorImage loadImage(final Bitmap bitmap){
        //load bitmap into tensorImage
        inputImageBuffer.load(bitmap);

        // create processor for the tensroflow
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreProcessorNormalizeOP())
                .build();

        return imageProcessor.process(inputImageBuffer);

    }


    //load tflite model
    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException{
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startToOffset = fileDescriptor.getStartOffset();
        long declareLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startToOffset, declareLength);
    }

    //normalize the image
    private TensorOperator getPreProcessorNormalizeOP(){
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    private TensorOperator getPostProcessorNormalizeOP(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    //create the barchart with the results
    public static void barChart(BarChart barChart, ArrayList<BarEntry> arrayList, final ArrayList<String> xAxisValues){

        barChart.setDrawBarShadow(false);
        barChart.setFitBars(true);
        barChart.setDrawValueAboveBar(true);
        barChart.setMaxVisibleValueCount(25);
        barChart.setPinchZoom(true);
        barChart.setDrawGridBackground(true);

        BarDataSet barDataSet = new BarDataSet(arrayList, "Class");
        // barDataSet.setColor(new int[]{Color.parseColor("#03A9F4"), Color.parseColor("#FF9800"),
        // Color.parseColor("#76FF03"), Color.parseColor("#E91E63"), Color.parseColor("#2962FF")});

        barDataSet.setColor(Color.parseColor("#03A9F4"));
        barDataSet.setColor(Color.parseColor("#FF9800"));
        barDataSet.setColor(Color.parseColor("#76FF03"));
        barDataSet.setColor(Color.parseColor("#E91E63"));
        barDataSet.setColor(Color.parseColor("#2962FF"));

        BarData barData = new BarData(barDataSet);
        barData.setBarWidth(0.9f);
        barData.setValueTextSize(0f);

        barChart.setBackgroundColor(Color.WHITE);
        barChart.setDrawGridBackground(false);
        barChart.animateY(2000);

        XAxis xAxis = barChart.getXAxis();
        xAxis.setTextSize(13f);
        xAxis.setTextColor(Color.BLACK);
        xAxis.setPosition(XAxis.XAxisPosition.TOP_INSIDE);
        xAxis.setValueFormatter(new IndexAxisValueFormatter(xAxisValues));
        xAxis.setDrawGridLines(false);

        barChart.setData(barData);

    }

    //showing result in barchart
    private void showResults(){
        try {
            labels = FileUtil.loadLabels(MainActivity.this, "labels.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }

        Map<String, Float> labelsProbability = new TensorLabel(labels,probabiltyProcessor.process(outputProbabilityBuffer))
                .getMapWithFloatValue();
        float maxValueinMap = (Collections.max(labelsProbability.values()));

        for(Map.Entry<String, Float> entry : labelsProbability.entrySet()){
            String[] label = labelsProbability.keySet().toArray(new String[0]);
            Float[] label_Probability = labelsProbability.values().toArray(new Float[0]);


            mbarChart = findViewById(R.id.chart);
            mbarChart.getXAxis().setDrawGridLines(false);
            mbarChart.getAxisLeft().setDrawGridLines(false);

            ArrayList<BarEntry> barEntries = new ArrayList<>();
            for(int i = 0; i < label_Probability.length; i++){
                barEntries.add(new BarEntry(i, label_Probability[i]*100));
            }

            ArrayList<String> xAxisName = new ArrayList<>();
            for(int i = 0; i <label.length; i++){
                xAxisName.add(label[i]);
            }
            barChart(mbarChart, barEntries, xAxisName);
            prediction.setText("Prediction");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 12 && resultCode == RESULT_OK && data != null){
            imageuri = data.getData();

            try{
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageuri);
                imageView.setImageBitmap(bitmap);
            }catch (IOException e){
                e.printStackTrace();
            }
        }
    }
}